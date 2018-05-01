//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "ComputationNode.h"
#include "Constants.h"
#include "Matrix.h"
#include "TensorView.h"
#include <unordered_set>
#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <list>
#include <memory>
#include <algorithm>
#include <utility>
#include <assert.h>
#include <set>
#include "Quantizers.h"
#include "InputAndParamNodes.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
// PlusNode (summand1, summand2)
// -----------------------------------------------------------------------

template <class ElemType>
class QuantizedProxyTimesNode : public ComputationNode<ElemType>, public NumInputs<4>
{
    typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"QuantizedProxyTimesNode"; }

public:
    DeclareConstructorFromConfigWithNumInputs(QuantizedProxyTimesNode);
    QuantizedProxyTimesNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {        
        NOT_IMPLEMENTED
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        NOT_IMPLEMENTED
    }
};

template class QuantizedProxyTimesNode<float>;
}}}
