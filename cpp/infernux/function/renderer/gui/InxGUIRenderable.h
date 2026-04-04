#pragma once

#include "InxGUIContext.h"

#include <core/log/InxLog.h>
#include <string>
#include <unordered_map>

namespace infernux
{
class InxGUIRenderable
{
  public:
    virtual ~InxGUIRenderable() = default;
    virtual void OnRender(InxGUIContext *ctx)
    {
        INXLOG_FATAL("InxGUIRenderable::OnRender not implemented");
    }

    /// Optional sub-timing breakdown (accumulated ms, reset by caller).
    virtual std::unordered_map<std::string, double> ConsumeSubTimings() { return {}; }
};
} // namespace infernux