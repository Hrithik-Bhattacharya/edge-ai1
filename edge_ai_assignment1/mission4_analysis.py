
# Mission 4: ESP32-S3 Architecture and Memory Analysis

SRAM_LIMIT = 300 * 1024
PSRAM_LIMIT = 8 * 1024 * 1024


def analyze_width(w):
    # Architecture: 5 -> w -> w -> 1
    layers = [(5, w), (w, w), (w, 1)]

    total_params = 0
    total_macs = 0

    for a, b in layers:
        # Each connection has one weight.
        # Each output neuron also has one bias.
        total_params += a * b + b

        # One multiply-accumulate per weight.
        total_macs += a * b

    # Memory required to store weights and biases
    bytes_f32 = total_params * 4
    bytes_int8 = total_params

    return {
        "width": w,
        "parameters": total_params,
        "macs": total_macs,
        "bytes_f32": bytes_f32,
        "bytes_int8": bytes_int8,
        "sram_f32": bytes_f32 <= SRAM_LIMIT,
        "sram_int8": bytes_int8 <= SRAM_LIMIT,
        "psram_f32": bytes_f32 <= PSRAM_LIMIT,
        "psram_int8": bytes_int8 <= PSRAM_LIMIT,
    }

def find_max_width(bytes_per_param):
    w = 1

    while True:
        result = analyze_width(w)
        required_bytes = result["parameters"] * bytes_per_param

        if required_bytes > SRAM_LIMIT:
            return w - 1

        w += 1


# ESP32-S3 INFERENCE SPEED ASSUMPTION
#
# Hardware reference:
# Espressif ESP32-S3 specifications:
# https://www.espressif.com/en/products/socs/esp32-s3
#
# The ESP32-S3 has a dual-core Xtensa LX7 processor operating
# at up to 240 MHz, with vector instructions for AI acceleration.
#
# Assumed effective throughput: 1,000,000 MACs/second.
#
# Justification:
# This is an illustrative, deliberately low throughput assumption.
# At 240 MHz, it corresponds to approximately 240 clock cycles
# per MAC. It is NOT a measured ESP32-S3 benchmark.
#
# Estimated inference time = (Total MACs / MACs per second) * 1000
#
# Limitations:
# The estimate excludes activation functions, memory transfers,
# quantization overhead and other runtime operations.
# Actual inference latency requires benchmarking on the ESP32-S3.

def estimate_inference_ms(macs, macs_per_second=1_000_000):
    return (macs / macs_per_second) * 1000



# Compare the five required architectures
widths = [32, 128, 256, 512, 2048]

print("MISSION 4 - ARCHITECTURE COMPARISON\n")

for w in widths:
    result = analyze_width(w)

    print(f"Architecture: 5 -> {w} -> {w} -> 1")
    print(f"Parameters: {result['parameters']}")
    print(f"MACs: {result['macs']}")
    print(f"Float32 bytes: {result['bytes_f32']}")
    print(f"Int8 bytes: {result['bytes_int8']}")
    print(f"Fits SRAM (Float32): {result['sram_f32']}")
    print(f"Fits SRAM (Int8): {result['sram_int8']}")
    print(f"Fits PSRAM (Float32): {result['psram_f32']}")
    print(f"Fits PSRAM (Int8): {result['psram_int8']}")
    print("-" * 45)

print("\nMISSION 4 - SRAM CUTOFF WIDTHS")

max_f32 = find_max_width(4)
max_int8 = find_max_width(1)

print(f"Maximum Float32 width: {max_f32}")
print(f"Maximum Int8 width: {max_int8}")

for w in [273, 274, 550, 551]:
    result = analyze_width(w)
    print(
        f"Width {w}: "
        f"Float32={result['bytes_f32']} bytes, "
        f"Int8={result['bytes_int8']} bytes"
    )

print("\nMISSION 4 - ESTIMATED ESP32-S3 INFERENCE TIME")
print("Assumed effective speed: 1,000,000 MACs/second")

for w in [16, 32, 128, 256, 512, 2048]:
    result = analyze_width(w)
    estimated_ms = estimate_inference_ms(result["macs"])

    print(f"Width {w}: {estimated_ms:.3f} ms")
