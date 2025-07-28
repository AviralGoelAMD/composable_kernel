# `make_naive_tensor_descriptor`

`make_naive_tensor_descriptor` is a utility function that creates a **tensor descriptor** describing the memory layout and access patterns for a multi-dimensional tensor. This descriptor is foundational for defining how tensors are stored and accessed in memory, enabling efficient and correct kernel implementations.

---

## **What is a Tensor Descriptor?**
A tensor descriptor is a metadata structure that describes:
- **Dimensions** (lengths) of the tensor
- **Memory layout** (strides) for accessing elements
- **Vectorization** capabilities for efficient memory access

---

## **Function Signature**
```cpp
template <typename... Lengths,
          typename... Strides,
          index_t GuaranteedLastDimensionVectorLength = -1,
          index_t GuaranteedLastDimensionVectorStride = -1,
          typename std::enable_if<sizeof...(Lengths) == sizeof...(Strides), bool>::type = false>
CK_TILE_HOST_DEVICE constexpr auto
make_naive_tensor_descriptor(const tuple<Lengths...>& lengths,
                             const tuple<Strides...>& strides,
                             number<GuaranteedLastDimensionVectorLength> = number<-1>{},
                             number<GuaranteedLastDimensionVectorStride> = number<-1>{});
```

---

## **Parameters**
- **`lengths`**: Tuple of tensor dimensions (e.g., `{M, N}` for an M×N matrix)
- **`strides`**: Tuple of memory strides for each dimension (e.g., `{N, 1}` for row-major layout)
- **`GuaranteedLastDimensionVectorLength`**: (Optional) Vector length for the last dimension (for vectorized access)
- **`GuaranteedLastDimensionVectorStride`**: (Optional) Stride for vectorized access in the last dimension

---

## **What It Does**
- **Creates a transform**: Maps logical tensor indices to physical memory addresses using the provided strides.
- **Sets up hidden/visible dimension IDs**: For internal bookkeeping of which dimensions are visible to the user and which are used for internal transformations.
- **Calculates element space size**: Determines the total number of elements (including any padding due to strides).
- **Handles vectorization**: Optionally encodes information about vectorized access in the last dimension (for efficient loads/stores).

---

## **Return Value**
Returns a `tensor_descriptor` object that:
- Encapsulates the shape and memory layout of your tensor
- Can be used to create tensor views, windows, and for efficient memory access in kernels
- Supports both scalar and vectorized access patterns

---

## **Usage Example**

### **1. 2D Matrix, Row-Major**
```cpp
// For a 2D matrix of shape (M, N), row-major
auto desc = make_naive_tensor_descriptor(
    make_tuple(M, N),      // lengths: M rows, N columns
    make_tuple(N, 1)       // strides: row-major layout
);
```

### **2. With Vectorization**
```cpp
// For a 2D matrix with vectorized access in the last dimension
auto desc = make_naive_tensor_descriptor(
    make_tuple(M, N),
    make_tuple(N, 1),
    number<Vector_N>{},    // vector length
    number<1>{}            // vector stride
);
```

---

## **Summary Table**
| Parameter | Description |
|-----------|-------------|
| `lengths` | Tensor dimensions (tuple) |
| `strides` | Memory strides (tuple) |
| `GuaranteedLastDimensionVectorLength` | Vector length for last dimension (optional) |
| `GuaranteedLastDimensionVectorStride` | Vector stride for last dimension (optional) |

---

---

## **What can you use a `naive_tensor_descriptor` for?**

### **1. Creating Tensor Views**
- **Function:** `make_tensor_view`
- **Usage:**  
  ```cpp
  auto tensor_view = make_tensor_view<address_space_enum::lds>(buffer_ptr, tensor_descriptor);
  ```
  This creates a view into a buffer (in global or LDS memory) with the shape and layout described by the descriptor.

**Function Signature:**
```cpp
template <address_space_enum BufferAddressSpace = address_space_enum::generic,
          amd_buffer_coherence_enum Coherence = amd_buffer_coherence_enum::coherence_default,
          typename DataType,
          typename... Ts>
CK_TILE_HOST_DEVICE constexpr auto make_tensor_view(DataType* p,
                                                    const tensor_descriptor<Ts...>& desc)
```

---

### **2. Creating Naive Tensor Views**
- **Function:** `make_naive_tensor_view`
- **Usage:**  
  ```cpp
  auto tensor_view = make_naive_tensor_view<address_space_enum::lds>(
      buffer_ptr, lengths_tuple, strides_tuple, vector_length, vector_stride);
  ```
  This creates a tensor view directly from lengths and strides, internally creating a naive tensor descriptor.

**Function Signature:**
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

### **3. Creating Tile Windows**
- **Function:** `make_tile_window`
- **Usage:**  
  ```cpp
  auto tile_window = make_tile_window(tensor_view, shape_tuple, origin_tuple, optional_distribution);
  ```
  Tile windows are used to define subregions (tiles) of a tensor for block-wise or tile-wise operations.

---

### **4. Transforming Descriptors**
- **Function:** `transform_tensor_descriptor`
- **Usage:**  
  ```cpp
  auto new_desc = transform_tensor_descriptor(old_tensor_descriptor, ...);
  ```
  This allows you to reshape, permute, or otherwise transform the logical view of a tensor.

---

### **5. Loading and Storing Tiles**
- **Functions:** `load_tile`, `store_tile`
- **Usage:**  
  ```cpp
  auto tile = load_tile(tile_window);
  store_tile(tile_window, tile);
  ```
  These functions use tile windows (which are built from tensor descriptors) to efficiently move data in and out of memory.

---

### **6. Host Tensor Construction**
- **Usage:**  
  Some host-side tensor classes (e.g., `HostTensor`) can be constructed using a tensor descriptor to define their shape and layout.

---

## **Summary Table**

| Function/Utility         | Purpose                                      | Expects Descriptor? |
|-------------------------|----------------------------------------------|---------------------|
| `make_tensor_view`      | Create a view into a buffer                  | Yes                |
| `make_naive_tensor_view` | Create a view with inline descriptor creation | No (creates internally) |
| `make_tile_window`      | Define a tile/subregion of a tensor          | Yes (indirectly)   |
| `transform_tensor_descriptor` | Transform/reshape tensor layout         | Yes                |
| `load_tile` / `store_tile` | Load/store data from/to a tile window      | Yes (via window)   |
| HostTensor (constructor) | Create host-side tensor with layout          | Yes                |

---

## **Summary**
- `make_naive_tensor_descriptor` builds a descriptor for a tensor with a straightforward, user-specified memory layout.
- It supports both scalar and vectorized access.
- It is foundational for defining how tensors are stored and accessed in memory for high-performance computing and GPU kernels. 