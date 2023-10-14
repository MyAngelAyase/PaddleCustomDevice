// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifdef PADDLE_WITH_ASCEND_TRANSFORMER_ACC
#include <acl/acl.h>
#include "atb_layer_base.h"
#include "kernels/funcs/format_utils.h"

void PpAscendAtbOpBase::BuildVariantPack(std::vector<const phi::DenseTensor *> &inTensors,
                                              std::vector<const phi::DenseTensor *> &outTensors)
{
  variantPacks_.inTensors.resize(inTensors.size());
  for (size_t i = 0; i < inTensors.size(); i++) {
    variantPacks_.inTensors.at(i) = ConvertDenseTensorToAtbTensor(*(inTensors.at(i)));
    if (variantPacks_.inTensors.at(i).desc.format == ACL_FORMAT_NCHW) {
      variantPacks_.inTensors.at(i).desc.format = ACL_FORMAT_ND;
    }
  }

  variantPacks_.outTensors.resize(outTensors.size());
  for (size_t i = 0; i < outTensors.size(); i++) {
    variantPacks_.outTensors.at(i) = ConvertDenseTensorToAtbTensor(*(outTensors.at(i)));
    if (variantPacks_.outTensors.at(i).desc.format == ACL_FORMAT_NCHW) {
      variantPacks_.outTensors.at(i).desc.format = ACL_FORMAT_ND;
    }
  }
}

void PpAscendAtbOpBase::SetWorkspace(uint64_t workspace_size)
{
  if (workspace_size <= workspaceSize_) {
    return;
  }

  if (workspace_) {
    aclrtFree(workspace_);
    workspace_ = nullptr;
    workspaceSize_ = 0;
  }
  int st = aclrtMalloc((void **)&workspace_, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
  PADDLE_ENFORCE_EQ(st,
                    0,
                    phi::errors::External("LayerOperation %s SetWorkspace MemMallocDevice,"
                            "fail, ret: %d size %llu.", opName_, st, workspace_size));

  workspaceSize_ = workspace_size;
}

atb::Status PpAscendAtbOpBase::Execute(aclrtStream stream,
                                     std::vector<const phi::DenseTensor *> &inTensors,
                                     std::vector<const phi::DenseTensor *> &outTensors)
{
  uint64_t workspace_size;
  stream_ = stream;
  BuildVariantPack(inTensors, outTensors);

  if(context_ == nullptr) {
    atb::CreateContext(&context_);
    context_->SetExecuteStream(stream);
  }

  atb::Status st = operation_->Setup(variantPacks_, workspace_size);
  PADDLE_ENFORCE_EQ(st,
                    0,
                    phi::errors::External("Atb Layer %s Op Setup failed,"
                                          "ret message: %d .", opName_, st));

  if (workspace_size > 0) {
    SetWorkspace(workspace_size);
  }

  st = operation_->Execute(variantPacks_, (uint8_t *)workspace_, workspace_size, context_);

  return st;
}

PpAscendAtbOpBase::PpAscendAtbOpBase(const std::string &opName)
{
  opName_ = opName;
}

PpAscendAtbOpBase::~PpAscendAtbOpBase() {}




void PpAscendAtbOpBaseAsync::BuildVariantPack(std::vector<const phi::DenseTensor *> &inTensors,
                                              std::vector<const phi::DenseTensor *> &outTensors,
                                              uint64_t layerId)
{
  variantPacks_.at(layerId).inTensors.resize(inTensors.size());
  for (size_t i = 0; i < inTensors.size(); i++) {
    variantPacks_.at(layerId).inTensors.at(i) = ConvertDenseTensorToAtbTensor(*(inTensors.at(i)));
    if (variantPacks_.at(layerId).inTensors.at(i).desc.format == ACL_FORMAT_NCHW) {
      variantPacks_.at(layerId).inTensors.at(i).desc.format = ACL_FORMAT_ND;
    }
  }

  variantPacks_.at(layerId).outTensors.resize(outTensors.size());
  for (size_t i = 0; i < outTensors.size(); i++) {
    variantPacks_.at(layerId).outTensors.at(i) = ConvertDenseTensorToAtbTensor(*(outTensors.at(i)));
    if (variantPacks_.at(layerId).outTensors.at(i).desc.format == ACL_FORMAT_NCHW) {
      variantPacks_.at(layerId).outTensors.at(i).desc.format = ACL_FORMAT_ND;
    }
  }
}

void PpAscendAtbOpBaseAsync::SetWorkspace(uint64_t workspace_size)
{
  if (workspace_size <= workspaceSize_) {
    return;
  }

  if (workspace_) {
    aclrtFree(workspace_);
    workspace_ = nullptr;
    workspaceSize_ = 0;
  }
  int st = aclrtMalloc((void **)&workspace_, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
  PADDLE_ENFORCE_EQ(st,
                    0,
                    phi::errors::External("AtbAsync layer %s SetWorkspace MemMallocDevice,"
                            "fail, ret: %d size %llu.", opName_, st, workspace_size));

  workspaceSize_ = workspace_size;
}

atb::Status PpAscendAtbOpBaseAsync::Setup(aclrtStream stream,
                                          std::vector<const phi::DenseTensor *> &inTensors,
                                          std::vector<const phi::DenseTensor *> &outTensors,
                                          uint64_t layerId)
{
  uint64_t workspace_size;
  stream_ = stream;
  BuildVariantPack(inTensors, outTensors, layerId);

  if (context_ == nullptr) {
    atb::CreateContext(&context_);
    context_->SetExecuteStream(stream);
  }

  atb::Status st = operations_.at(layerId)->Setup(variantPacks_.at(layerId), workspace_size);
  PADDLE_ENFORCE_EQ(st,
                    0,
                    phi::errors::External("AtbAsync Layer %s Op Setup failed,"
                                          "ret message: %d .", opName_, st));

  if (workspace_size > 0) {
    SetWorkspace(workspace_size);
  }
  PushTask(layerId);

  return st;
}

atb::Status PpAscendAtbOpBaseAsync::Execute(uint64_t layerId)
{
  atb::Status st = operations_.at(layerId)->Execute(variantPacks_.at(layerId), (uint8_t *)workspace_, workspaceSize_, context_);
  PADDLE_ENFORCE_EQ(st,
                    0,
                    phi::errors::External("AtbAsync Layer %dth Execute failed,"
                                          "ret message: %s .", layerId, st));

  return st;
}

void PpAscendAtbOpBaseAsync::PushTask(int layerId)
{
  std::unique_lock<std::mutex> lock(mutex_);
  taskQueue_.push(layerId);
  lock.unlock();
  cond_.notify_one();
}

int PpAscendAtbOpBaseAsync::PopTask()
{
  std::unique_lock<std::mutex> lock(mutex_);
  while (taskQueue_.empty()) {
      cond_.wait(lock);
  }
  int layerId = taskQueue_.front();
  taskQueue_.pop();
  return layerId;
}

void PpAscendAtbOpBaseAsync::ThreadProcessTask()
{
  // ATB_LOG(FATAL) << "PpAscendAtbOpBaseAsync ThreadProcessTask start";

  int ret = aclrtSetDevice(currentDevId_);
  PADDLE_ENFORCE_EQ(ret,
                    0,
                    phi::errors::External("AtbAsync Layer %s Op SetDevice failed,"
                                          "ret message: %d .", opName_, ret));

  int processTaskCount = 0;
  while (true) {
    int layerId = PopTask();

    atb::Status st = Execute(layerId);

    processTaskCount++;
    if (processTaskCount == layerNum_) {
        // ATB_LOG(INFO) << "PpAscendAtbOpBaseAsync thread process all layers";
        processTaskCount = 0;
        allTaskFinish_ = true;
    }
  }
}

PpAscendAtbOpBaseAsync::PpAscendAtbOpBaseAsync(const std::string &opName)
{
  opName_ = opName;
}

PpAscendAtbOpBaseAsync::~PpAscendAtbOpBaseAsync() {}
#endif
