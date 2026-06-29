import struct

TAG_TO_BITS = {
    'FLAT': 0b00,
    'EDGE': 0b01,
    'MOTION': 0b10,
    'CHAOS': 0b11
}

BITS_TO_TAG = {v: k for k, v in TAG_TO_BITS.items()}

class BitWriter:
    """Handles sub-byte serialization for the Quadtree Header"""
    def __init__(self):
        self.data = bytearray()
        self.accumulator = 0
        self.bit_count = 0

    def write(self, value, num_bits):
        for i in range(num_bits - 1, -1, -1):
            bit = (value >> i) & 1
            self.accumulator = (self.accumulator << 1) | bit
            self.bit_count += 1
            if self.bit_count == 8:
                self.data.append(self.accumulator)
                self.accumulator = 0
                self.bit_count = 0

    def flush(self):
        if self.bit_count > 0:
            self.accumulator <<= (8 - self.bit_count)
            self.data.append(self.accumulator)
            self.accumulator = 0
            self.bit_count = 0
        return bytes(self.data)

class BitReader:
    """Handles sub-byte deserialization"""
    def __init__(self, data_bytes):
        self.data = data_bytes
        self.byte_pos = 0
        self.bit_pos = 7

    def read(self, num_bits):
        value = 0
        for _ in range(num_bits):
            if self.byte_pos >= len(self.data):
                raise EOFError("BitReader out of bounds")
            
            bit = (self.data[self.byte_pos] >> self.bit_pos) & 1
            value = (value << 1) | bit
            
            self.bit_pos -= 1
            if self.bit_pos < 0:
                self.bit_pos = 7
                self.byte_pos += 1
        return value

def serialize_quadtree(node, writer):
    """Recursively packs the Quadtree into bits"""
    if not node.is_leaf:
        writer.write(1, 1)  # 1 bit: SPLIT
        for child in node.children:
            serialize_quadtree(child, writer)
    else:
        writer.write(0, 1)  # 1 bit: LEAF
        tag_val = TAG_TO_BITS.get(node.tag, 0)
        writer.write(tag_val, 2)  # 2 bits: SEMANTIC TAG

def deserialize_quadtree(reader, x, y, size, min_size=8):
    """Recursively reconstructs the Quadtree from bits"""
    from lab_v6 import QuadNode  # Assuming QuadNode is in lab_v6.py
    
    node = QuadNode(x, y, size)
    is_split = reader.read(1)
    
    if is_split == 1:
        node.is_leaf = False
        half = size // 2
        node.children.append(deserialize_quadtree(reader, x, y, half, min_size))
        node.children.append(deserialize_quadtree(reader, x + half, y, half, min_size))
        node.children.append(deserialize_quadtree(reader, x, y + half, half, min_size))
        node.children.append(deserialize_quadtree(reader, x + half, y + half, half, min_size))
    else:
        tag_val = reader.read(2)
        node.tag = BITS_TO_TAG.get(tag_val, 'FLAT')
        
    return node

# --- TEST HARNESS ---
if __name__ == "__main__":
    from lab_v6 import QuadNode
    
    # Manually construct a mock quadtree
    # Root (64x64) -> Split
    root = QuadNode(0, 0, 64)
    root.is_leaf = False
    
    # Child 1 (FLAT), Child 2 (EDGE), Child 3 (MOTION)
    c1 = QuadNode(0, 0, 32); c1.tag = 'FLAT'
    c2 = QuadNode(32, 0, 32); c2.tag = 'EDGE'
    c3 = QuadNode(0, 32, 32); c3.tag = 'MOTION'
    
    # Child 4 -> Split into CHAOS and FLAT
    c4 = QuadNode(32, 32, 32)
    c4.is_leaf = False
    c4_1 = QuadNode(32, 32, 16); c4_1.tag = 'CHAOS'
    c4_2 = QuadNode(48, 32, 16); c4_2.tag = 'FLAT'
    c4_3 = QuadNode(32, 48, 16); c4_3.tag = 'FLAT'
    c4_4 = QuadNode(48, 48, 16); c4_4.tag = 'FLAT'
    c4.children = [c4_1, c4_2, c4_3, c4_4]
    
    root.children = [c1, c2, c3, c4]
    
    # Serialize
    writer = BitWriter()
    serialize_quadtree(root, writer)
    packed_bytes = writer.flush()
    
    print(f"Packed Tree Size: {len(packed_bytes)} bytes")
    print(f"Packed Bits: {bin(int.from_bytes(packed_bytes, 'big'))}")