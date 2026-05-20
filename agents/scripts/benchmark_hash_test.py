import sys
try:
    import atom_rust
    print(atom_rust.compute_string_hash("hello world"))
except Exception as e:
    print(f"Failed: {e}")
