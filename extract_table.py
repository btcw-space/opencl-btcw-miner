#!/usr/bin/env python3
"""
Extract the secp256k1_ecmult_gen_prec_table from kernel.ptx into a binary file.
This is a one-time extraction script.

The table is 32768 bytes (128 groups x 4 entries x 64 bytes per affine point).
Format: secp256k1_ge_storage = {x[4 x uint64_LE], y[4 x uint64_LE]}
"""

import re
import sys
import os

def extract_table(ptx_path, out_path):
    with open(ptx_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if 'secp256k1_ecmult_gen_prec_table[32768]' in line:
                # Extract the byte values between { and }
                m = re.search(r'\{([^}]+)\}', line)
                if not m:
                    print("ERROR: Found table declaration but couldn't parse byte values")
                    sys.exit(1)

                byte_str = m.group(1)
                byte_vals = [int(x.strip()) for x in byte_str.split(',')]

                if len(byte_vals) != 32768:
                    print(f"ERROR: Expected 32768 bytes, got {len(byte_vals)}")
                    sys.exit(1)

                data = bytes(byte_vals)

                with open(out_path, 'wb') as out:
                    out.write(data)

                print(f"Extracted {len(data)} bytes to {out_path}")

                # Verify first point (should be identity / zero for group 0, value 0)
                first_64 = data[:64]
                print(f"First entry (group 0, value 0): {first_64.hex()}")

                # Second entry (group 0, value 1 = 1*G)
                second_64 = data[64:128]
                print(f"Second entry (group 0, value 1 = G): {second_64.hex()}")

                # Known generator x-coordinate (little-endian limb order):
                # Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
                # In LE limbs: {0x59F2815B16F81798, 0x029BFCDB2DCE28D9, 0x55A06295CE870B07, 0x79BE667EF9DCBBAC}
                # As bytes (LE): 98 17 F8 16 5B 81 F2 59  D9 28 CE 2D DB FC 9B 02 ...
                gx_le = bytes([
                    0x98, 0x17, 0xF8, 0x16, 0x5B, 0x81, 0xF2, 0x59,
                    0xD9, 0x28, 0xCE, 0x2D, 0xDB, 0xFC, 0x9B, 0x02,
                    0x07, 0x0B, 0x87, 0xCE, 0x95, 0x62, 0xA0, 0x55,
                    0xAC, 0xBB, 0xDC, 0xF9, 0x7E, 0x66, 0xBE, 0x79
                ])
                if second_64[:32] == gx_le:
                    print("VERIFIED: Entry 1 x-coordinate matches generator point G!")
                else:
                    print(f"WARNING: Entry 1 x-coord doesn't match G directly.")
                    print(f"  Expected: {gx_le.hex()}")
                    print(f"  Got:      {second_64[:32].hex()}")
                    print("  (This may indicate a different table layout or normalization)")

                return True

    print("ERROR: Could not find secp256k1_ecmult_gen_prec_table in PTX file")
    sys.exit(1)

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ptx_path = os.path.join(script_dir, '..', 'kernel.ptx')
    out_path = os.path.join(script_dir, 'ecmult_gen_table.bin')
    extract_table(ptx_path, out_path)
