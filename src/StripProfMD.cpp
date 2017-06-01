//===-- PGOInstrumentation.cpp - MST-based PGO Instrumentation ------------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

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
#include "llvm/IR/Function.h"
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

namespace {

class StripProfMetadataPass : public ModulePass {
public:
  static char ID;
  StripProfMetadataPass() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    bool updated = false;

    if(M.getNamedMetadata(Twine("llvm.module.flags")) && M.getMaximumFunctionCount()!=None ){
       M.eraseNamedMetadata( M.getNamedMetadata(Twine("llvm.module.flags")) );
       updated = true;
    }
    for(auto F = M.begin(); F!=M.end(); F++){
       if(F->getMetadata(llvm::LLVMContext::MD_prof)!=nullptr){
          F->setMetadata(llvm::LLVMContext::MD_prof, nullptr);
          updated = true;
       }

       for(auto bb = F->begin(); bb!=F->end(); bb++){
          Instruction *TI = bb->getTerminator();
          if(TI==nullptr)continue;

          if(TI->getMetadata(llvm::LLVMContext::MD_prof)!=nullptr){
             TI->setMetadata(llvm::LLVMContext::MD_prof, nullptr);
             updated = true;
          }
       }
    }
    return updated;
  }
};

}

char StripProfMetadataPass::ID = 0;
static RegisterPass<StripProfMetadataPass> X("strip-prof-md",
                      "Remove all profiling metadata.", false, false);

