//===-- PGOInstrumentation.cpp - MST-based PGO Instrumentation ------------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements work relaxed instrumentation as described in the paper:
//   [1] Rodrigo C. O. Rocha, Pavlos Petoumenos, Luis F. W. Goes, Murray Cole,
//   Zheng Wang, Hugh Leather. <<NAME OF PAPER>>. 
// This instrumentation algorithm is based on the optimal instrumentation
// proposed in the following papers:
// on the following paper:
//   [2] Donald E. Knuth, Francis R. Stevenson. Optimal measurement of points
//   for program frequency counts. BIT Numerical Mathematics 1973, Volume 13,
//   Issue 3, pp 313-322
//   [3] Thomas Ball and James Larus. Optimally profiling and tracing programs.
//   ACM Transactions on Programming Languages and Systems (TOPLAS) 16.4 (1994):
//   1319-1360.
//
// The idea of the algorithm based on the fact that for each node (except for
// the entry and exit), the sum of incoming edge counts equals the sum of
// outgoing edge counts. The count of edge on spanning tree can be derived from
// those edges not on the spanning tree. Knuth proves this method instruments
// the minimum number of edges.
//
// The minimal spanning tree here is actually a maximum weight tree -- on-tree
// edges have higher frequencies (more likely to execute). The idea is to
// instrument those less frequently executed edges to reduce the runtime
// overhead of instrumented binaries.
//
// This file contains the Pass WorkInstrumentationGen which work-based relaxed
// instrumentation that instruments the IR to generate counters to compute the
// amount of work performed during the program's execution.
//
// Class PGOEdge represents a CFG edge and some auxiliary information. Class
// BBInfo contains auxiliary information for each BB. These two classes are used
// in pass PGOInstrumentationGen. Class PGOUseEdge and UseBBInfo are the derived
// class of PGOEdge and BBInfo, respectively. They contains extra data structure
// used in populating profile counters.
// The MST implementation is in Class CFGMST (CFGMST.h).
//
//===----------------------------------------------------------------------===//

#include "WorkInstrumentation.h"
#include "CFGMST.h"
#include "CostModel.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JamCRC.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cstdint>
#include <map>
#include <list>
#include <queue>
#include <math.h>

using namespace llvm;

#define DEBUG_TYPE "pgo-instrumentation"

STATISTIC(NumOfPGOInstrument, "Number of edges instrumented.");
STATISTIC(NumOfPGOEdge, "Number of edges.");
STATISTIC(NumOfPGOBB, "Number of basic-blocks.");
STATISTIC(NumOfPGOSplit, "Number of critical edge splits.");

static cl::opt<unsigned> MaxErrorAllowed(
    "work-instr-error-threshold", cl::init(0), cl::Hidden, cl::ZeroOrMore,
    cl::desc("Max percentage error allowed for the relaxed profiling of the work metric"));

static cl::opt<bool>
WorkAvgError("work-instr-avg-error", cl::init(false), cl::Hidden,
	      cl::desc("Use the percentage error based on the average work per DAG"));

static cl::opt<unsigned> GreedySolverThreshold(
    "work-instr-greedy-solver", cl::init(15), cl::Hidden, cl::ZeroOrMore,
    cl::desc("Threshold for using the brute force solver or the greedy solver"));

namespace {

/// \brief An MST based instrumentation for PGO
///
/// Implements a Minimum Spanning Tree (MST) based instrumentation for PGO
/// in the function level.
struct PGOEdge {
  // This class implements the CFG edges. Note the CFG can be a multi-graph.
  // So there might be multiple edges with same SrcBB and DestBB.
  const BasicBlock *SrcBB;
  const BasicBlock *DestBB;
  uint64_t Weight;
  bool InMST;
  bool Removed;
  bool IsCritical;
  PGOEdge(const BasicBlock *Src, const BasicBlock *Dest, unsigned W = 1)
      : SrcBB(Src), DestBB(Dest), Weight(W), InMST(false), Removed(false),
        IsCritical(false) {}
  // Return the information string of an edge.
  const std::string infoString() const {
    return (Twine(Removed ? "-" : " ") + (InMST ? " " : "*") +
            (IsCritical ? "c" : " ") + "  W=" + Twine(Weight)).str();
  }
};

// This class stores the auxiliary information for each BB.
struct BBInfo {
  BBInfo *Group;
  uint32_t Index;
  uint32_t Rank;

  BBInfo(unsigned IX) : Group(this), Index(IX), Rank(0) {}

  // Return the information string of this object.
  const std::string infoString() const {
    return (Twine("Index=") + Twine(Index)).str();
  }
};

// This class implements the CFG edges. Note the CFG can be a multi-graph.
template <class Edge, class BBInfo> class FuncPGOInstrumentation {
private:
  Function &F;

public:
  std::string FuncName;


  // The Minimum Spanning Tree of function CFG.
  CFGMST<Edge, BBInfo> MST;

  // Give an edge, find the BB that will be instrumented.
  // Return nullptr if there is no BB to be instrumented.
  BasicBlock *getInstrBB(Edge *E);

  // Return the auxiliary BB information.
  BBInfo &getBBInfo(const BasicBlock *BB) const { return MST.getBBInfo(BB); }

  // Dump edges and BB information.
  void dumpInfo(std::string Str = "") const {
    MST.dumpEdges(dbgs(), Twine("Dump Function ") + FuncName + "\t" + Str);
  }

  FuncPGOInstrumentation(
      Function &Func,
      BranchProbabilityInfo *BPI = nullptr,
      BlockFrequencyInfo *BFI = nullptr)
      : F(Func), MST(F, BPI, BFI) {


    FuncName = std::string(F.getName());

    DEBUG(dumpInfo("after CFGMST"));

    NumOfPGOBB += MST.BBInfos.size();
    for (auto &E : MST.AllEdges) {
      if (E->Removed)
        continue;
      NumOfPGOEdge++;
      if (!E->InMST)
        NumOfPGOInstrument++;
    }
  }

  // Return the number of profile counters needed for the function.
  unsigned getNumCounters() {
    unsigned NumCounters = 0;
    for (auto &E : this->MST.AllEdges) {
      if (!E->InMST && !E->Removed)
        NumCounters++;
    }
    return NumCounters;
  }
};

// Given a CFG E to be instrumented, find which BB to place the instrumented
// code. The function will split the critical edge if necessary.
template <class Edge, class BBInfo>
BasicBlock *FuncPGOInstrumentation<Edge, BBInfo>::getInstrBB(Edge *E) {
  if(E->InMST || E->Removed)
    return nullptr;

  BasicBlock *SrcBB = const_cast<BasicBlock *>(E->SrcBB);
  BasicBlock *DestBB = const_cast<BasicBlock *>(E->DestBB);
  // For a fake edge, instrument the real BB.
  if(SrcBB == nullptr)
    return DestBB;
  if(DestBB == nullptr)
    return SrcBB;

  // Instrument the SrcBB if it has a single successor,
  // otherwise, the DestBB if this is not a critical edge.
  TerminatorInst *TI = SrcBB->getTerminator();
  if(TI->getNumSuccessors() <= 1)
    return SrcBB;
  if(!E->IsCritical)
    return DestBB;

  // For a critical edge, we have to split. Instrument the newly
  // created BB.
  NumOfPGOSplit++;
  DEBUG(dbgs() << "Split critical edge: " << getBBInfo(SrcBB).Index << " --> "
               << getBBInfo(DestBB).Index << "\n");
  unsigned SuccNum = GetSuccessorNumber(SrcBB, DestBB);
  BasicBlock *InstrBB = SplitCriticalEdge(TI, SuccNum);
  assert(InstrBB && "Critical edge is not split");

  E->Removed = true;
  return InstrBB;
}

static std::string getVertexName(const BasicBlock *bb){
   if(bb){
      std::string name;
      raw_string_ostream namestream(name);
      bb->printAsOperand(namestream,false);
      return namestream.str();
   }else return "[nullptr]";
}

class InstrBBSet {
public:
  bool known = false;
  std::set<const BasicBlock *> incSet;
  std::set<const BasicBlock *> decSet;

  void addInc(const BasicBlock *bb){
     this->incSet.insert(bb);
  }

  void addInc(const std::set<const BasicBlock *> &incList){
     for(const BasicBlock *bb : incList){
        this->incSet.insert(bb);
     }
  }

  void addDec(const std::set<const BasicBlock *> &decList){
     for(const BasicBlock *bb : decList){
        this->decSet.insert(bb);
     }
  }

  void addDec(const BasicBlock * bb){
     this->decSet.insert(bb);
  }

  void add(InstrBBSet &bbSet){
     this->addInc(bbSet.incSet);
     this->addDec(bbSet.decSet);
  }

  void sub(InstrBBSet &bbSet){
     this->addInc(bbSet.decSet);
     this->addDec(bbSet.incSet);
  }

  void simplify(){
     std::set<const BasicBlock *> tmpIncSet = this->incSet;
     std::set<const BasicBlock *> tmpDecSet = this->decSet;
     for(const BasicBlock *positiveBB : tmpIncSet){
      if(tmpDecSet.find(positiveBB)!=tmpDecSet.end())
         this->incSet.erase(positiveBB);
      }
      for(const BasicBlock *negativeBB : tmpDecSet){
         if(tmpIncSet.find(negativeBB)!=tmpIncSet.end())
            this->decSet.erase(negativeBB);
      }
  }

  void dump(){
     errs() << "\tInc:";
     for(const BasicBlock *bb : this->incSet){
        errs() << " " << getVertexName(bb);
     }
     errs() << "\n\tDec:";
     for(const BasicBlock *bb : this->decSet){
        errs() << " " << getVertexName(bb);
     }
     errs() << "\n";
  }

  int computeCost(std::map<const BasicBlock *,int> bbCosts){
     int cost = 0;
     for(const BasicBlock *bb : this->incSet){
        cost += bbCosts[bb];
     }

     for(const BasicBlock *bb : this->decSet){
        cost -= bbCosts[bb];
     }

     return cost;
  }
};


InstrBBSet computeBBSets(FuncPGOInstrumentation<PGOEdge, BBInfo> &FuncInfo, std::map<const BasicBlock *, std::map<const BasicBlock *, InstrBBSet> > &edgeBBSets, const BasicBlock *bb){
   InstrBBSet newBBSet;
   
   for(const BasicBlock *pred : FuncInfo.MST.getPredecessors(bb)){
      if(edgeBBSets[pred][bb].known){
         newBBSet.add(edgeBBSets[pred][bb]);
      }
   }
   
   for(const BasicBlock *succ : FuncInfo.MST.getSuccessors(bb)){
      if(edgeBBSets[bb][succ].known){
         newBBSet.add(edgeBBSets[bb][succ]);
      }
   }
   //cancelling out increment with decrement counting sets
   newBBSet.simplify();
   return newBBSet;
}

void populateEdgeBBSets(FuncPGOInstrumentation<PGOEdge, BBInfo> &FuncInfo, std::map<const BasicBlock *, std::map<const BasicBlock *, InstrBBSet> > &edgeBBSets){
   bool changed = true;
   while(changed){
      changed = false;
      for(const BasicBlock *bb : FuncInfo.MST.AllBasicBlocks){
         unsigned unknownInSets = 0;
         unsigned unknownOutSets = 0;
         for(const BasicBlock *pred : FuncInfo.MST.getPredecessors(bb)){
            if(!edgeBBSets[pred][bb].known) unknownInSets++;
         }
         for(const BasicBlock *succ : FuncInfo.MST.getSuccessors(bb)){
            if(!edgeBBSets[bb][succ].known) unknownOutSets++;
         }
         if(unknownInSets==0 && unknownOutSets==1){
            changed = true;
            InstrBBSet newBBSet;
            newBBSet.known = true;
            for(const BasicBlock *pred : FuncInfo.MST.getPredecessors(bb)){
               newBBSet.add(edgeBBSets[pred][bb]);
            }
            for(const BasicBlock *succ : FuncInfo.MST.getSuccessors(bb)){
               if(edgeBBSets[bb][succ].known){
                  newBBSet.sub(edgeBBSets[bb][succ]);
               }
            }
            //cancelling out increment with decrement counting sets
            newBBSet.simplify();
            //computing instrumentation set for unknown edge
            for(const BasicBlock *succ : FuncInfo.MST.getSuccessors(bb)){
               if(!edgeBBSets[bb][succ].known){
                  edgeBBSets[bb][succ] = newBBSet;
                  edgeBBSets[bb][succ].known = true;
                  unknownOutSets--;
               }
            }
         }
         if(unknownInSets==1 && unknownOutSets==0){
            changed = true;
            InstrBBSet newBBSet;
            newBBSet.known = true;
            for(const BasicBlock *pred : FuncInfo.MST.getPredecessors(bb)){
               if(edgeBBSets[pred][bb].known){
                  newBBSet.sub(edgeBBSets[pred][bb]);
               }
            }
            for(const BasicBlock *succ : FuncInfo.MST.getSuccessors(bb)){
               newBBSet.add(edgeBBSets[bb][succ]);
            }
            //cancelling out increment with decrement counting sets
            newBBSet.simplify();
            //computing instrumentation set for unknown edge
            for(const BasicBlock *pred : FuncInfo.MST.getPredecessors(bb)){
               if(!edgeBBSets[pred][bb].known){
                  edgeBBSets[pred][bb] = newBBSet;
                  edgeBBSets[pred][bb].known = true;
                  unknownInSets--;
               }
            }
         }
      }
   }
}

// Returns the maximum value that can be put in a knapsack of capacity W
unsigned knapsackBrute(double maxError, unsigned idx, std::vector<BasicBlock*> &bbs, std::map<BasicBlock *,double> &errors, std::map<BasicBlock *,unsigned> &freq, std::map<BasicBlock *,bool> &removed){
   // Base Case
   if(idx >= bbs.size() || maxError <= 0)
      return 0;

   // If weight of the nth item is more than Knapsack capacity W, then
   // this item cannot be included in the optimal solution
   if(errors[bbs[idx]] > maxError){
      unsigned score = knapsackBrute(maxError, idx+1, bbs, errors, freq, removed);
      removed[bbs[idx]] = false;
      return score;
   }
   // Return the maximum of two cases: (1) nth item included (2) not included
   else{
      std::map<BasicBlock *,bool> removedWithout = removed;
      std::map<BasicBlock *,bool> removedWith = removed;
      unsigned scoreWithout = knapsackBrute(maxError, idx+1, bbs, errors, freq, removedWithout);
      unsigned scoreWith = freq[bbs[idx]] + knapsackBrute(maxError-errors[bbs[idx]], idx+1, bbs, errors, freq, removedWith);
      if(scoreWithout>scoreWith){
         removed = removedWithout;
         removed[bbs[idx]] = false;
         return scoreWithout;
      }else{
         removed = removedWith;
         removed[bbs[idx]] = true;
         return scoreWith;
      }
    }
}


// Returns the maximum value that can be put in a knapsack of capacity W
unsigned knapsackGreedy(double maxError, std::vector<BasicBlock*> &bbs, std::map<BasicBlock *,double> &errors, std::map<BasicBlock *,unsigned> &freq, std::map<BasicBlock *,bool> &removed){
   //    sorting Item on basis of ration
   std::sort(bbs.begin(), bbs.end(),
             [&freq,&errors](BasicBlock *bb1, BasicBlock *bb2){
                     double r1 = double(freq[bb1])/errors[bb1];
                     double r2 = double(freq[bb2])/errors[bb2];
                     return r1>r2;
             }

   );

   double curError = 0.0;  // Current error in knapsack
   unsigned finalvalue = 0; // Result (value in Knapsack)
 
   // Looping through all Items
   for(unsigned i = 0; i < bbs.size(); i++){
      // If adding Item won't overflow, add it completely
      if(curError + errors[bbs[i]] <= maxError){
         curError += errors[bbs[i]];
         finalvalue += freq[bbs[i]];
         removed[bbs[i]] = true;
      }else removed[bbs[i]] = false;
   }
 
   return finalvalue;
}

unsigned knapsack(double maxError, std::vector<BasicBlock*> &bbs, std::map<BasicBlock *,double> &errors, std::map<BasicBlock *,unsigned> &freq, std::map<BasicBlock *,bool> &removed){
   if(bbs.size()>GreedySolverThreshold){
      errs() << "Executing the greedy knapsack: " << bbs.size() << "\n";
      return knapsackGreedy(maxError, bbs, errors, freq, removed);
   }else {
      errs() << "Executing the brute-force knapsack: " << bbs.size() << "\n";
      return knapsackBrute(maxError, 0, bbs, errors, freq, removed);
   }
}


int maxCost(BasicBlock *bb, Loop *loop, std::map<const BasicBlock *,int> &bbCosts, LoopInfo *loopInfo, std::map<BasicBlock*,bool> &visited){
   BasicBlock *loopHeader = nullptr;
   if(loop) loopHeader = loop->getHeader();
   visited[bb] = true;

   if(bb!=loopHeader && loopInfo->isLoopHeader(bb)){
      Loop *innerLoop = loopInfo->getLoopFor(bb);

      //SmallVector<BasicBlock*,8> exits;
      //innerLoop->getExitBlocks(exits);
      //if(exits.size()==1){
      BasicBlock *succ = innerLoop->getExitBlock();
      if(succ){
         if(!visited[succ])
            return bbCosts[bb]+maxCost(succ,loop,bbCosts,loopInfo,visited);
      }
      return bbCosts[bb];
   }else{
      bool hasSuccessor = false;
      int max = INT_MIN;
      for(succ_iterator sit = succ_begin(bb), end = succ_end(bb); sit!=end; sit++){
         BasicBlock *succ = dyn_cast<BasicBlock>(*sit);
         if(!visited[succ]){
            bool loopContains = true;
            if(loop) loopContains = loop->contains(succ);
            if(loopContains){
               hasSuccessor = true;
               std::map<BasicBlock*,bool> newVisited = visited;
               int val = maxCost(succ,loop,bbCosts,loopInfo,newVisited);
               if(val>max) {
                  max=val;
                  visited = newVisited;
               }
            }
         }
      }
      if(hasSuccessor) return bbCosts[bb]+max;
      else return bbCosts[bb];
   }
}

int maxCost(BasicBlock *bb, Loop *loop, std::map<const BasicBlock *,int> &bbCosts, LoopInfo *loopInfo){
   std::map<BasicBlock*,bool> visited;
   return maxCost(bb,loop,bbCosts,loopInfo,visited);
}

int minCost(BasicBlock *bb, Loop *loop, std::map<const BasicBlock *,int> &bbCosts, LoopInfo *loopInfo, std::map<BasicBlock*,bool> &visited){
   BasicBlock *loopHeader = nullptr;
   if(loop) loopHeader = loop->getHeader();
   visited[bb] = true;

   if(bb!=loopHeader && loopInfo->isLoopHeader(bb)){
      Loop *innerLoop = loopInfo->getLoopFor(bb);

      //SmallVector<BasicBlock*,8> exits;
      //innerLoop->getExitBlocks(exits);
      //if(exits.size()==1){
      BasicBlock *succ = innerLoop->getExitBlock();
      if(succ){
         if(!visited[succ])
            return bbCosts[bb]+minCost(succ,loop,bbCosts,loopInfo,visited);
      }
      return bbCosts[bb];
   }else{
      bool hasSuccessor = false;
      int min = INT_MAX;
      for(succ_iterator sit = succ_begin(bb), end = succ_end(bb); sit!=end; sit++){
         BasicBlock *succ = dyn_cast<BasicBlock>(*sit);
         if(!visited[succ]){
            bool loopContains = true;
            if(loop) loopContains = loop->contains(succ);
            if(loopContains){
               hasSuccessor = true;
               std::map<BasicBlock*,bool> newVisited = visited;
               int val = minCost(succ,loop,bbCosts,loopInfo,newVisited);
               if(val<min){
                  min=val;
                  visited = newVisited;
               }
            }
         }
      }
      if(hasSuccessor) return bbCosts[bb]+min;
      else return bbCosts[bb];
   }
}

int minCost(BasicBlock *bb, Loop *loop, std::map<const BasicBlock *,int> &bbCosts, LoopInfo *loopInfo){
   std::map<BasicBlock*,bool> visited;
   return minCost(bb,loop,bbCosts,loopInfo,visited);
}

void relaxInstrDAG(BasicBlock *entry, Loop *loop, std::map<const BasicBlock *,int> &bbCosts, std::map<const BasicBlock *,bool> &instrBBs, std::map<const BasicBlock *, InstrBBSet> &instrBBSets, LoopInfo *loopInfo, BlockFrequencyInfo *BFI){
   BasicBlock *loopHeader = nullptr;
   if(loop) loopHeader = loop->getHeader();

   std::queue<BasicBlock*> q;
   std::set<BasicBlock*> s;
   q.push(entry);
   s.insert(entry);
   while(!q.empty()){
      BasicBlock *bb = q.front();
      q.pop();
      errs() << "\t"<<getVertexName(bb)<<"\n";
      if(bb!=loopHeader && loopInfo->isLoopHeader(bb)){
         Loop *innerLoop = loopInfo->getLoopFor(bb);
         BasicBlock *succ = innerLoop->getExitBlock();
         if(succ && s.find(succ)==s.end()){
            q.push(succ);
            s.insert(succ);
         }
      }else{
         for(succ_iterator sit = succ_begin(bb), end = succ_end(bb); sit!=end; sit++){
            BasicBlock *succ = dyn_cast<BasicBlock>(*sit);
            bool loopContains = true;
            if(loop) loopContains = loop->contains(succ);
            if(s.find(succ)==s.end() && loopContains){
               q.push(succ);
               s.insert(succ);
            }
         }
      }
   }
   errs() << "Computing the min cost\n";
   int minVal = minCost(entry,loop,bbCosts,loopInfo);
   errs() << "Min Cost: " << minVal << "\n";

   int maxVal;
   if(WorkAvgError){
      errs() << "Computing the max cost\n";
      maxVal = maxCost(entry,loop,bbCosts,loopInfo);
      errs() << "Max Cost: " << maxVal << "\n";
   }
   std::vector<BasicBlock*> bbs;
   std::map<BasicBlock *,double> errors;
   std::map<BasicBlock *,unsigned> freq;
   std::map<BasicBlock *,bool> removed;



   for(BasicBlock *bb : s){
      if(instrBBs[bb]){
	 errs () << "Processing probe in " << getVertexName(bb) << "\n";
         int probeVal = (instrBBSets[bb].computeCost(bbCosts));
         double percentError;
         if(WorkAvgError)
            percentError = double(abs(probeVal))/double(abs(maxVal-minVal)*0.5);
         else percentError = fabs(probeVal)/fabs(minVal);
         errs() << "Percentage error by removing " << getVertexName(bb) << ": " << percentError << " : freq " << BFI->getBlockFreq(bb).getFrequency() << "\n";
         errs() << "\tProbe value: " << probeVal << "\n";
         bbs.push_back(bb);
         freq[bb] = BFI->getBlockFreq(bb).getFrequency();
         errors[bb] = percentError;
         /*
         if(percentError<0.01){
            instrBBs[bb] = false;
         }
         */
      }
   }
   
   unsigned score = knapsack(double(MaxErrorAllowed)/100.0, bbs, errors, freq, removed);
   errs() << "Knapsack score: " << score << "\n";
   errs() << "Removed:";
   double error = 0;
   for(BasicBlock *bb : bbs){
      if(removed[bb]){
         errs() << " " << getVertexName(bb);
         error += errors[bb];
         instrBBs[bb] = false;
      }
   }
   errs() << "\n";
   errs() << "Error: " << error << "\n";
}

void performRelaxation(Function &F, std::map<const BasicBlock *,int> &bbCosts,
 std::map<const BasicBlock *,bool> &instrBBs,
 std::map<const BasicBlock *, InstrBBSet> &instrBBSets,
 //ScalarEvolution *SE,
 LoopInfo *loopInfo, BlockFrequencyInfo *BFI){
   //Break if no relaxation is allowed
   if(MaxErrorAllowed==0)return;
   errs() << "Dumping loops:\n";
   for(auto bb = F.begin(); bb!=F.end(); bb++){
      if(loopInfo->isLoopHeader(&(*bb))){
         Loop *loop = loopInfo->getLoopFor(&(*bb));
         loop->print(errs());
         /*
         errs() << "Trip Count: " << SE->getSmallConstantTripCount(loop) << "\n";
         SE->getBackedgeTakenCount(loop)->dump();
         //SE->getUnsignedRange(SE->getBackedgeTakenCount(loop)).dump();
         SE->getMaxBackedgeTakenCount(loop)->dump();
         //SE->getUnsignedRange(SE->getMaxBackedgeTakenCount(loop)).dump();
         auto *phiNode = loop->getCanonicalInductionVariable();
         if(phiNode) phiNode->dump();
         else errs() << "Could not find canonical ind-var\n";
         */
         errs() << "Relaxing loop:\n";
         relaxInstrDAG(loop->getHeader(),loop,bbCosts,instrBBs,instrBBSets,loopInfo,BFI);
      }
   }
  
   errs() << "Outer DAG in function\n";
   relaxInstrDAG(&F.getEntryBlock(),nullptr,bbCosts,instrBBs,instrBBSets,loopInfo,BFI);
}

// Visit all edge and instrument the edges not in MST, and do value profiling.
// Critical edges will be split.
static void instrumentOneFunc(Function &F, Module *M, TargetTransformInfo *TTI,
 BranchProbabilityInfo *BPI, BlockFrequencyInfo *BFI, LoopInfo *loopInfo) {
   errs() << "Instrumenting function: " << F.getName() << "\n";
   FuncPGOInstrumentation<PGOEdge, BBInfo> FuncInfo(F, BPI, BFI);

   Value *localCounterPtr = nullptr;
   Instruction *entryBBInsertingPt = nullptr;
   
   std::map<const BasicBlock *,int> bbCosts;
   std::map<const BasicBlock *,bool> instrBBs;
   std::map<const BasicBlock *, std::map<const BasicBlock *, InstrBBSet> > edgeBBSets;
   std::map<const BasicBlock *, InstrBBSet> instrBBSets;

   for(auto &E : FuncInfo.MST.AllEdges){
      edgeBBSets[E->SrcBB][E->DestBB].known = false;
   }
   
   unsigned numOfBBs = 0;
   unsigned numOfInstrBBs = 0;
   /*
   unsigned NumCounters = FuncInfo.getNumCounters();
   errs() << "Num Counters "<< NumCounters<< "\n";
   */

   errs() << "BB costs:\n";
   bbCosts[nullptr] = 0;
   for(auto bbit = F.begin(); bbit!=F.end(); bbit++){
      BasicBlock *bb = &(*bbit);
      numOfBBs++;
      bbCosts[bb] = getBasicBlockCost(TTI, bb);
      instrBBs[bb] = false;
      errs() << getVertexName(bb) << " : " << bbCosts[bb] << "\n";
   }

   errs() << "NumOfBBs: " << numOfBBs << "\n";

   for(auto &E : FuncInfo.MST.AllEdges){
      //errs() << "getInstrBB\n";
      errs() << "Edge: " << ((E->SrcBB)?(E->SrcBB->getName()):"") << ":" << reinterpret_cast<std::uintptr_t>(E->SrcBB) << " -> " << ((E->DestBB)?(E->DestBB->getName()):"") << ":"<<reinterpret_cast<std::uintptr_t>(E->DestBB) << "\n";
      BasicBlock *InstrBB = FuncInfo.getInstrBB(E.get());
      if(E->Removed) errs() << "\tRemoved\n";
      if(E->InMST) errs() << "\tInMST\n";
      if(!InstrBB){
         errs() << "\tSkipping\n";
         continue;
      }
      edgeBBSets[(BasicBlock*)(E->SrcBB)][(BasicBlock*)(E->DestBB)].known = true;
      edgeBBSets[(BasicBlock*)(E->SrcBB)][(BasicBlock*)(E->DestBB)].addInc(InstrBB);

      errs() << "\tInstrumenting: "<< InstrBB->getName() <<":"<<reinterpret_cast<std::uintptr_t>(InstrBB) <<"\n";
      //if(!instrBBs[InstrBB]) numOfInstrBBs++;
      instrBBs[InstrBB] = true;
   }

   //errs() << "NumOfInstrBBs: " << numOfInstrBBs << "\n";

   //compute instrumentation
   populateEdgeBBSets(FuncInfo, edgeBBSets);
   for(const BasicBlock *bb : FuncInfo.MST.AllBasicBlocks){
      errs() << "Instrumentation set for: " << getVertexName(bb) << "\n";
      InstrBBSet bbSet = computeBBSets(FuncInfo, edgeBBSets, bb);
      bbSet.dump();
      for(const BasicBlock *instrBB : bbSet.incSet){
         instrBBSets[instrBB].addInc(bb);
      }
      for(const BasicBlock *instrBB : bbSet.decSet){
         instrBBSets[instrBB].addDec(bb);
      }
   }

   errs() << "Performing relaxation:\n";
   performRelaxation(F, bbCosts, instrBBs, instrBBSets, loopInfo, BFI);


   //perform actual instrumentation
   errs() << "Creating local counter: \n";
   entryBBInsertingPt = static_cast<Instruction*>( &(*F.begin()->getFirstInsertionPt()));
   IRBuilder<> builder(entryBBInsertingPt);
   localCounterPtr = builder.CreateAlloca(builder.getInt64Ty());
   builder.CreateStore(builder.getInt64(0), localCounterPtr);

   numOfInstrBBs = 0;
   for(auto bb = F.begin(); bb!=F.end(); bb++){
      BasicBlock *InstrBB = static_cast<BasicBlock *>(&(*bb));
      if(!instrBBs[InstrBB]) continue;

      //double pathCost = computePathCost(InstrBB,pred,succ,FuncInfo,bbCosts);
      //double pathCost = weights[InstrBB]+bbCosts[InstrBB];
      int pathCost = instrBBSets[InstrBB].computeCost(bbCosts);
      if(pathCost==0) continue; //skip instrumentation with no contribution

      numOfInstrBBs++;

      errs() << "Perform instrumentation in basic block: "<< getVertexName(InstrBB) << " [" << pathCost <<"]\n";
      instrBBSets[InstrBB].dump();

      Instruction *insertingPt = static_cast<Instruction*>( &(*InstrBB->getFirstInsertionPt()) );
      if(InstrBB==(&F.getEntryBlock())){//if entry basic block
         insertingPt = entryBBInsertingPt;
      }

      IRBuilder<> Builder(insertingPt);
      assert(Builder.GetInsertPoint() != InstrBB->end() &&
           "Cannot get the Instrumentation point");


      TerminatorInst *tinstr = InstrBB->getTerminator();
      if(tinstr->getOpcode()==Instruction::Ret){
         Value *updatedLocalCounter = Builder.CreateAdd(Builder.CreateLoad(localCounterPtr), Builder.getInt64(pathCost));
         Value *globalCounter = M->getOrInsertGlobal("__work_counter", Builder.getInt64Ty());
         Value *globalCounterPtr = Builder.CreateLoad(globalCounter);
         Value *updatedGlobalCounter = Builder.CreateAdd(updatedLocalCounter, globalCounterPtr);
         Builder.CreateStore(updatedGlobalCounter, globalCounter);
      }else{
         Builder.CreateStore(Builder.CreateAdd(Builder.CreateLoad(localCounterPtr), Builder.getInt64(pathCost)), localCounterPtr);
      }
   }

   errs() << "NumOfInstrBBs: " << numOfInstrBBs << "\n";

   for(auto bb = F.begin(); bb!=F.end(); bb++){
      TerminatorInst *tinstr = bb->getTerminator();
      if(tinstr->getOpcode()==Instruction::Ret){
         IRBuilder<> Builder(tinstr);
         if(!instrBBs[&(*bb)]){
            errs() << "Recording local counter to global counter\n";
            Value *globalCounter = M->getOrInsertGlobal("__work_counter", Builder.getInt64Ty());
            Value *globalCounterPtr = Builder.CreateLoad(globalCounter);
      
            Value *updatedGlobalCounter = Builder.CreateAdd(Builder.CreateLoad(localCounterPtr), globalCounterPtr);
            Builder.CreateStore(updatedGlobalCounter, globalCounter);
         }
      }
   }

}

} // end anonymous namespace

static bool InstrumentAllFunctions(
 Module &M,
 function_ref<TargetTransformInfo *(Function &)> LookupTTI,
 function_ref<BranchProbabilityInfo *(Function &)> LookupBPI,
 function_ref<BlockFrequencyInfo *(Function &)> LookupBFI,
 //function_ref<ScalarEvolution *(Function &)> LookupSE,
 function_ref<LoopInfo *(Function &)> LookupLoopInfo) {
   for (auto &F : M) {
      if (F.isDeclaration()) continue;
      auto *BPI = LookupBPI(F);
      auto *BFI = LookupBFI(F);
      auto *TTI = LookupTTI(F);
      //RegionInfo *regionInfo = LookupRegionInfo(F);
      LoopInfo *loopInfo = LookupLoopInfo(F);
      //ScalarEvolution *SE = LookupSE(F);
      instrumentOneFunc(F, &M, TTI, BPI, BFI, loopInfo);
   }
   return true;
}

bool WorkInstrumentationGen::runOnModule(Module &M) {
   if (skipModule(M))
      return false;
   auto LookupTTI = [this](Function &F) {
      return &this->getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
   };
   auto LookupLoopInfo = [this](Function &F) {
      return &this->getAnalysis<LoopInfoWrapperPass>(F).getLoopInfo();
   };
   /*
   auto LookupSE = [this](Function &F) {
      return &this->getAnalysis<ScalarEvolutionWrapperPass>(F).getSE();
   };
   */
   auto LookupBPI = [this](Function &F) {
      return &this->getAnalysis<BranchProbabilityInfoWrapperPass>(F).getBPI();
   };
   auto LookupBFI = [this](Function &F) {
      return &this->getAnalysis<BlockFrequencyInfoWrapperPass>(F).getBFI();
   };
   return InstrumentAllFunctions(M, LookupTTI, LookupBPI, LookupBFI, LookupLoopInfo);
}

void WorkInstrumentationGen::getAnalysisUsage(AnalysisUsage &AU) const {
   AU.addRequired<BlockFrequencyInfoWrapperPass>();
   AU.addRequired<LoopInfoWrapperPass>();
   //AU.addRequired<ScalarEvolutionWrapperPass>();
   AU.addRequired<TargetTransformInfoWrapperPass>();
}

char WorkInstrumentationGen::ID = 0;
static RegisterPass<WorkInstrumentationGen> X("work-instr-gen",
                      "Work instrumentation.", false, false);

