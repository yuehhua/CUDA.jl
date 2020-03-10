# Julia wrapper for header: cutensor.h
# Automatically generated using Clang.jl


@checked function cutensorInit(handle)
    initialize_api()
    @runtime_ccall((:cutensorInit, libcutensor()), cutensorStatus_t,
                   (Ptr{cutensorHandle_t},),
                   handle)
end

@checked function cutensorInitTensorDescriptor(handle, desc, numModes, extent, stride,
                                               dataType, unaryOp)
    initialize_api()
    @runtime_ccall((:cutensorInitTensorDescriptor, libcutensor()), cutensorStatus_t,
                   (Ptr{cutensorHandle_t}, Ptr{cutensorTensorDescriptor_t}, UInt32,
                    Ptr{Int64}, Ptr{Int64}, cudaDataType_t, cutensorOperator_t),
                   handle, desc, numModes, extent, stride, dataType, unaryOp)
end

@checked function cutensorElementwiseTrinary(handle, alpha, A, descA, modeA, beta, B,
                                             descB, modeB, gamma, C, descC, modeC, D,
                                             descD, modeD, opAB, opABC, typeScalar, stream)
    initialize_api()
    @runtime_ccall((:cutensorElementwiseTrinary, libcutensor()), cutensorStatus_t,
                   (Ptr{cutensorHandle_t}, Ptr{Cvoid}, CuPtr{Cvoid},
                    Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, Ptr{Cvoid}, CuPtr{Cvoid},
                    Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, Ptr{Cvoid}, CuPtr{Cvoid},
                    Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, CuPtr{Cvoid},
                    Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, cutensorOperator_t,
                    cutensorOperator_t, cudaDataType_t, CUstream),
                   handle, alpha, A, descA, modeA, beta, B, descB, modeB, gamma, C, descC,
                   modeC, D, descD, modeD, opAB, opABC, typeScalar, stream)
end

@checked function cutensorElementwiseBinary(handle, alpha, A, descA, modeA, gamma, C,
                                            descC, modeC, D, descD, modeD, opAC,
                                            typeScalar, stream)
    initialize_api()
    @runtime_ccall((:cutensorElementwiseBinary, libcutensor()), cutensorStatus_t,
                   (Ptr{cutensorHandle_t}, Ptr{Cvoid}, CuPtr{Cvoid},
                    Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, Ptr{Cvoid}, CuPtr{Cvoid},
                    Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, CuPtr{Cvoid},
                    Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, cutensorOperator_t,
                    cudaDataType_t, CUstream),
                   handle, alpha, A, descA, modeA, gamma, C, descC, modeC, D, descD, modeD,
                   opAC, typeScalar, stream)
end

@checked function cutensorPermutation(handle, alpha, A, descA, modeA, B, descB, modeB,
                                      typeScalar, stream)
    initialize_api()
    @runtime_ccall((:cutensorPermutation, libcutensor()), cutensorStatus_t,
                   (Ptr{cutensorHandle_t}, Ptr{Cvoid}, CuPtr{Cvoid},
                    Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, CuPtr{Cvoid},
                    Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, cudaDataType_t, CUstream),
                   handle, alpha, A, descA, modeA, B, descB, modeB, typeScalar, stream)
end

@checked function cutensorInitContractionDescriptor(handle, desc, descA, modeA,
                                                    alignmentRequirementA, descB, modeB,
                                                    alignmentRequirementB, descC, modeC,
                                                    alignmentRequirementC, descD, modeD,
                                                    alignmentRequirementD, computeType)
    initialize_api()
    @runtime_ccall((:cutensorInitContractionDescriptor, libcutensor()), cutensorStatus_t,
                   (Ptr{cutensorHandle_t}, Ptr{cutensorContractionDescriptor_t},
                    Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, UInt32,
                    Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, UInt32,
                    Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, UInt32,
                    Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, UInt32,
                    cutensorComputeType_t),
                   handle, desc, descA, modeA, alignmentRequirementA, descB, modeB,
                   alignmentRequirementB, descC, modeC, alignmentRequirementC, descD,
                   modeD, alignmentRequirementD, computeType)
end

@checked function cutensorInitContractionFind(handle, find, algo)
    initialize_api()
    @runtime_ccall((:cutensorInitContractionFind, libcutensor()), cutensorStatus_t,
                   (Ptr{cutensorHandle_t}, Ptr{cutensorContractionFind_t}, cutensorAlgo_t),
                   handle, find, algo)
end

@checked function cutensorContractionGetWorkspace(handle, desc, find, pref, workspaceSize)
    initialize_api()
    @runtime_ccall((:cutensorContractionGetWorkspace, libcutensor()), cutensorStatus_t,
                   (Ptr{cutensorHandle_t}, Ptr{cutensorContractionDescriptor_t},
                    Ptr{cutensorContractionFind_t}, cutensorWorksizePreference_t,
                    Ptr{UInt64}),
                   handle, desc, find, pref, workspaceSize)
end

@checked function cutensorInitContractionPlan(handle, plan, desc, find, workspaceSize)
    initialize_api()
    @runtime_ccall((:cutensorInitContractionPlan, libcutensor()), cutensorStatus_t,
                   (Ptr{cutensorHandle_t}, Ptr{cutensorContractionPlan_t},
                    Ptr{cutensorContractionDescriptor_t}, Ptr{cutensorContractionFind_t},
                    UInt64),
                   handle, plan, desc, find, workspaceSize)
end

@checked function cutensorContraction(handle, plan, alpha, A, B, beta, C, D, workspace,
                                      workspaceSize, stream)
    initialize_api()
    @runtime_ccall((:cutensorContraction, libcutensor()), cutensorStatus_t,
                   (Ptr{cutensorHandle_t}, Ptr{cutensorContractionPlan_t}, Ptr{Cvoid},
                    CuPtr{Cvoid}, CuPtr{Cvoid}, Ptr{Cvoid}, CuPtr{Cvoid}, CuPtr{Cvoid},
                    CuPtr{Cvoid}, UInt64, CUstream),
                   handle, plan, alpha, A, B, beta, C, D, workspace, workspaceSize, stream)
end

@checked function cutensorContractionMaxAlgos(maxNumAlgos)
    initialize_api()
    @runtime_ccall((:cutensorContractionMaxAlgos, libcutensor()), cutensorStatus_t,
                   (Ptr{Int32},),
                   maxNumAlgos)
end

@checked function cutensorReduction(handle, alpha, A, descA, modeA, beta, C, descC, modeC,
                                    D, descD, modeD, opReduce, minTypeCompute, workspace,
                                    workspaceSize, stream)
    initialize_api()
    @runtime_ccall((:cutensorReduction, libcutensor()), cutensorStatus_t,
                   (Ptr{cutensorHandle_t}, Ptr{Cvoid}, CuPtr{Cvoid},
                    Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, Ptr{Cvoid}, CuPtr{Cvoid},
                    Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, CuPtr{Cvoid},
                    Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, cutensorOperator_t,
                    cutensorComputeType_t, CuPtr{Cvoid}, UInt64, CUstream),
                   handle, alpha, A, descA, modeA, beta, C, descC, modeC, D, descD, modeD,
                   opReduce, minTypeCompute, workspace, workspaceSize, stream)
end

@checked function cutensorReductionGetWorkspace(handle, A, descA_, modeA, C, descC_, modeC,
                                                D, descD_, modeD, opReduce, typeCompute,
                                                workspaceSize)
    initialize_api()
    @runtime_ccall((:cutensorReductionGetWorkspace, libcutensor()), cutensorStatus_t,
                   (Ptr{cutensorHandle_t}, CuPtr{Cvoid}, Ptr{cutensorTensorDescriptor_t},
                    Ptr{Int32}, CuPtr{Cvoid}, Ptr{cutensorTensorDescriptor_t}, Ptr{Int32},
                    CuPtr{Cvoid}, Ptr{cutensorTensorDescriptor_t}, Ptr{Int32},
                    cutensorOperator_t, cutensorComputeType_t, Ptr{UInt64}),
                   handle, A, descA_, modeA, C, descC_, modeC, D, descD_, modeD, opReduce,
                   typeCompute, workspaceSize)
end

@checked function cutensorGetAlignmentRequirement(handle, ptr, desc, alignmentRequirement)
    initialize_api()
    @runtime_ccall((:cutensorGetAlignmentRequirement, libcutensor()), cutensorStatus_t,
                   (Ptr{cutensorHandle_t}, CuPtr{Cvoid}, Ptr{cutensorTensorDescriptor_t},
                    Ptr{UInt32}),
                   handle, ptr, desc, alignmentRequirement)
end

function cutensorGetErrorString(error)
    @runtime_ccall((:cutensorGetErrorString, libcutensor()), Cstring,
                   (cutensorStatus_t,),
                   error)
end

function cutensorGetVersion()
    @runtime_ccall((:cutensorGetVersion, libcutensor()), Csize_t, ())
end

function cutensorGetCudartVersion()
    @runtime_ccall((:cutensorGetCudartVersion, libcutensor()), Csize_t, ())
end
