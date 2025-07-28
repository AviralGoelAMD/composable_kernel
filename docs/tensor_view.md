# `tensor_view`

A **tensor view** is a complete tensor object that combines **metadata** (tensor descriptor) with **actual data** (buffer pointer) to provide a unified interface for accessing and manipulating multi-dimensional data in memory.

---

## **What is a Tensor View?**

A tensor view is a high-level abstraction that:
- **Encapsulates both metadata and data** in a single object
- **Provides unified access** to tensor elements regardless of memory space
- **Supports vectorized operations** for efficient memory access
- **Enables tile-based operations** for GPU kernels

---

## **Core Components**

A tensor view consists of two main parts:

### **1. Tensor Descriptor (Metadata)**
- **Shape and dimensions** of the tensor
- **Memory layout** (strides, padding)
- **Vectorization** capabilities
- **Access patterns** and transformations

### **2. Buffer View (Data)**
- **Pointer to actual data** in memory
- **Memory space information** (global, LDS, etc.)
- **Buffer coherence** settings
- **Memory operation** types

---

## **Function Signatures**

### **Creating Tensor Views**

#### **`make_tensor_view`**
```cpp
template <address_space_enum BufferAddressSpace = address_space_enum::generic,
          amd_buffer_coherence_enum Coherence = amd_buffer_coherence_enum::coherence_default,
          typename DataType,
          typename... Ts>
CK_TILE_HOST_DEVICE constexpr auto make_tensor_view(DataType* p,
                                                    const tensor_descriptor<Ts...>& desc)
```

#### **`make_naive_tensor_view`**
```cpp
template <address_space_enum BufferAddressSpace = address_space_enum::generic,
          memory_operation_enum DstInMemOp = memory_operation_enum::set,
          amd_buffer_coherence_enum Coherence = amd_buffer_coherence_enum::coherence_default,
          typename DataType,
          typename... Lengths,
          typename... Strides,
          index_t GuaranteedLastDimensionVectorLength = -1,
          index_t GuaranteedLastDimensionVectorStride = -1>
CK_TILE_HOST_DEVICE constexpr auto make_naive_tensor_view(DataType* p,
                                                         const tuple<Lengths...>& lengths,
                                                         const tuple<Strides...>& strides,
                                                         number<GuaranteedLastDimensionVectorLength> = number<-1>{},
                                                         number<GuaranteedLastDimensionVectorStride> = number<-1>{})
```

---

## **Usage Examples**

### **1. Creating Views from Descriptors**
```cpp
// Step 1: Create descriptor (metadata only)
auto desc = make_naive_tensor_descriptor(
    make_tuple(M, N),      // shape
    make_tuple(N, 1)       // strides
);

// Step 2: Create view (descriptor + data)
auto view = make_tensor_view<address_space_enum::lds>(
    buffer_ptr,  // data pointer
    desc         // metadata
);
```

### **2. Creating Views Directly**
```cpp
// Create view directly from lengths and strides
auto view = make_naive_tensor_view<address_space_enum::global>(
    global_buffer_ptr,
    make_tuple(M, N),      // lengths
    make_tuple(N, 1),      // strides
    number<Vector_N>{},    // vector length
    number<1>{}            // vector stride
);
```

### **3. LDS Buffer Views**
```cpp
// Common pattern in GPU kernels
__shared__ DataType lds_buffer[M * N];

auto lds_view = make_tensor_view<address_space_enum::lds>(
    lds_buffer,
    make_naive_tensor_descriptor(
        make_tuple(M, N),
        make_tuple(N, 1)
    )
);
```

---

## **Memory Spaces**

Tensor views support different memory spaces:

```cpp
// Global memory
auto global_view = make_tensor_view<address_space_enum::global>(ptr, desc);

// Local Data Store (LDS)
auto lds_view = make_tensor_view<address_space_enum::lds>(ptr, desc);

// Generic (default)
auto generic_view = make_tensor_view<address_space_enum::generic>(ptr, desc);
```

---

## **Vectorization Support**

Tensor views support vectorized access for efficient memory operations:

```cpp
// Vectorized access in last dimension
auto view = make_naive_tensor_view<address_space_enum::global>(
    ptr,
    make_tuple(M, N),
    make_tuple(N, 1),
    number<4>{},    // 4 elements per vector
    number<1>{}     // stride of 1
);

// Access 4 elements at once
auto vec_data = view.get_vectorized_elements(coord, offset);
```

---

## **Tile Window Creation**

Tensor views are used to create tile windows for block-wise operations:

```cpp
// Create tile window from view
auto tile_window = make_tile_window(
    view,                    // tensor view
    make_tuple(Tile_M, Tile_N),  // tile shape
    make_tuple(offset_M, offset_N),  // origin
    distribution             // optional distribution
);
```

---

## **Transformation Support**

Tensor views can be transformed to create new views with different layouts:

```cpp
// Transform view to new layout
auto new_view = transform_tensor_view(
    old_view,
    new_transforms,
    lower_dims,
    upper_dims
);
```

---

## **Common Patterns in GPU Kernels**

### **1. LDS Buffering Pattern**
```cpp
// Allocate LDS buffer
__shared__ DataType lds_buffer[Block_M * Block_N];

// Create LDS view
auto lds_view = make_tensor_view<address_space_enum::lds>(
    lds_buffer,
    make_naive_tensor_descriptor(
        make_tuple(Block_M, Block_N),
        make_tuple(Block_N, 1)
    )
);

// Create tile windows
auto lds_write_window = make_tile_window(lds_view, shape, origin);
auto lds_read_window = make_tile_window(lds_view, shape, origin, distribution);
```

### **2. Global Memory Access**
```cpp
// Create global memory view
auto global_view = make_naive_tensor_view<address_space_enum::global>(
    global_ptr,
    make_tuple(M, N),
    make_tuple(N, 1),
    number<Vector_N>{},
    number<1>{}
);

// Create tile window for block processing
auto global_window = make_tile_window(
    global_view,
    make_tuple(Block_M, Block_N),
    make_tuple(block_offset_M, block_offset_N),
    distribution
);
```

---

## **Tensor View vs Tensor Descriptor**


### **Relationship**
```cpp
// Tensor Descriptor = Metadata only
auto desc = make_naive_tensor_descriptor(lengths, strides);

// Tensor View = Descriptor + Data Pointer
auto view = make_tensor_view<address_space_enum::lds>(buffer_ptr, desc);
```

### **Analogy**
- **Tensor Descriptor** = **Blueprint** (shows how to build a house)
- **Tensor View** = **Actual House** (blueprint + materials + location)

### **When to Use Each**

**Use Tensor Descriptor when:**
- Defining tensor layouts and shapes
- Creating multiple views with same layout
- Transforming tensor layouts

**Use Tensor View when:**
- Accessing actual data
- Creating tile windows
- Performing memory operations


---

## **Summary**

- **Tensor View** is a complete tensor object combining metadata and data
- **Provides unified interface** for accessing tensor elements
- **Supports vectorized operations** for efficient memory access
- **Different from descriptor**: View has data pointer, descriptor is metadata only