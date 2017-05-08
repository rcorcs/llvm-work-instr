//===---- Transforms/WorkInstrumentation.h - Work instr pass ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file provides the interface for IR based instrumentation pass for
/// work profiling (work-instr-gen).
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_WORK_INSTRUMENTATION_H
#define LLVM_TRANSFORMS_WORK_INSTRUMENTATION_H


#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/IR/PassManager.h"
#include "llvm/Analysis/Passes.h"

#include "llvm/Transforms/Instrumentation.h"

using namespace llvm;

namespace {

/// The instrumentation (profile-instr-gen) pass for IR based PGO.
class WorkInstrumentationGen : public ModulePass {
public:
  static char ID;

  WorkInstrumentationGen() : ModulePass(ID) {}

  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  //PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // End llvm namespace
#endif
