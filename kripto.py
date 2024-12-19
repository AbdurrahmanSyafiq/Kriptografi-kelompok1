import streamlit as st
import pandas as pd
import numpy as np

# Fungsi untuk Walsh Transform
def walsh_transform(f):
    n = len(f).bit_length() - 1
    W = np.array(f) * 2 - 1
    for i in range(n):
        step = 2**(i + 1)
        for j in range(0, len(f), step):
            for k in range(2**i):
                a, b = W[j + k], W[j + k + 2**i]
                W[j + k], W[j + k + 2**i] = a + b, a - b
    return W

# Fungsi untuk Nonlinearity
def nonlinearity(sbox, n, m):
    table = []
    for output_bit in range(m):
        column = []
        for x in range(2**n):
            fx = (sbox[x] >> output_bit) & 1
            column.append(fx)
        table.append(column)
    min_distance = float('inf')
    for column in table:
        W = walsh_transform(column)
        max_walsh = np.max(np.abs(W))
        distance = 2**(n - 1) - max_walsh / 2
        if distance < min_distance:
            min_distance = distance
    return min_distance

# Fungsi untuk SAC
def compute_sac_matrix(sbox, n):
    sac_matrix = np.zeros((n, n))
    for input_bit in range(n):
        for output_bit in range(n):
            total_flips = 0
            for x in range(2**n):
                flipped_x = x ^ (1 << input_bit)
                original_output = (sbox[x] >> output_bit) & 1
                flipped_output = (sbox[flipped_x] >> output_bit) & 1
                total_flips += original_output ^ flipped_output
            sac_matrix[input_bit][output_bit] = total_flips / (2 ** n)
    return sac_matrix

# Fungsi untuk BIC-NL
def bic_nl_matrix(sboxes, n):
    num_sboxes = len(sboxes)
    bic_nl_values = np.zeros((num_sboxes, num_sboxes))
    for i in range(num_sboxes):
        for j in range(num_sboxes):
            if i == j:
                bic_nl_values[i, j] = np.nan
            else:
                total_nl = 0
                count = 0
                for bit1 in range(n):
                    for bit2 in range(bit1 + 1, n):
                        count += 1
                        sbox1_bit1 = [(y >> bit1) & 1 for y in sboxes[i]]
                        sbox2_bit2 = [(y >> bit2) & 1 for y in sboxes[j]]
                        combined = [b1 ^ b2 for b1, b2 in zip(sbox1_bit1, sbox2_bit2)]
                        W = walsh_transform(combined)
                        max_walsh = np.max(np.abs(W))
                        nl = 2**(n - 1) - max_walsh / 2
                        total_nl += nl
                bic_nl_values[i, j] = total_nl / count
    return bic_nl_values

# Fungsi untuk BIC-SAC
def calculate_sac(sbox, bit_i, bit_j):
    bit_length = 8
    n = len(sbox)
    independence_sum = 0

    for x in range(n):
        for bit_to_flip in range(bit_length):
            flipped_x = x ^ (1 << bit_to_flip)
            y1 = sbox[x]
            y2 = sbox[flipped_x]

            b1_i = (y1 >> bit_i) & 1
            b1_j = (y1 >> bit_j) & 1
            b2_i = (y2 >> bit_i) & 1
            b2_j = (y2 >> bit_j) & 1

            independence_sum += ((b1_i ^ b2_i) ^ (b1_j ^ b2_j))

    return independence_sum / (n * bit_length)

def generate_bic_sac_matrix(sbox):
    bit_length = 8
    matrix = np.zeros((bit_length, bit_length))

    for i in range(bit_length):
        for j in range(bit_length):
            if i != j:
                matrix[i][j] = calculate_sac(sbox, i, j)

    return matrix

# Fungsi untuk LAP
def lap(sbox, n):
    max_bias = 0
    result_data = []

    for a in range(1, 2**n):
        for b in range(1, 2**n):
            bias = 0

            for x in range(2**n):
                input_parity = bin(x & a).count('1') % 2
                output_parity = bin(sbox[x] & b).count('1') % 2

                if input_parity == output_parity:
                    bias += 1

            bias = abs(bias - 2**(n - 1)) / 2**n
            result_data.append({"a": a, "b": b, "Bias": bias})
            max_bias = max(max_bias, bias)

    return max_bias, result_data

# Fungsi untuk DAP
def dap(sbox, n):
    max_diff_prob = 0
    results = []

    for dx in range(1, 2**n):
        for dy in range(1, 2**n):
            count = 0

            for x in range(2**n):
                if sbox[x ^ dx] ^ sbox[x] == dy:
                    count += 1

            prob = count / 2**n
            max_diff_prob = max(max_diff_prob, prob)
            results.append({"Δx": dx, "Δy": dy, "Probabilitas": prob})

    return max_diff_prob, results

# Judul aplikasi
st.title("🔐 S-Box Metrics Calculator")
st.markdown("Use this tool to calculate various cryptographic metrics for S-Boxes.")

# Upload file CSV
st.sidebar.header("⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader("Upload CSV file for S-Box configuration", type=["csv"], help="Upload a CSV file containing the S-Box configuration.")

if uploaded_file:
    # Baca file CSV menjadi DataFrame
    df = pd.read_csv(uploaded_file, header=None)
    st.subheader("📄 Uploaded S-Box Configuration")
    st.dataframe(df, use_container_width=True)

    # Konversi DataFrame ke list
    sbox = df.values.flatten().tolist()
    st.write("**S-Box (List):**", sbox)

    # Validasi panjang S-box
    if len(sbox) != 256:
        st.error("Error: S-box harus memiliki panjang 256 elemen.")
    else:
        # Pilihan metrik
        metric = st.sidebar.selectbox("📊 Choose Metric:", ["Nonlinearity", "SAC", "BIC-NL", "BIC-SAC", "LAP", "DAP"], help="Select the cryptographic metric to calculate.")

        if metric == "Nonlinearity":
            st.subheader("Nonlinearity Calculation")
            n, m = 8, 8
            nl_value = nonlinearity(sbox, n, m)
            st.success(f"**Nonlinearity Value:** {nl_value}")

        elif metric == "SAC":
            st.subheader("Strict Avalanche Criterion (SAC)")
            n = 8
            sac_matrix = compute_sac_matrix(sbox, n)
            df_sac = pd.DataFrame(sac_matrix, columns=[f"Bit-{i+1}" for i in range(n)], index=[f"Bit-{i+1}" for i in range(n)])
            st.write("**SAC Matrix:**")
            st.dataframe(df_sac, use_container_width=True)

            excel_filename = "sac_matrix.xlsx"
            df_sac.to_excel(excel_filename, index=True, sheet_name="SAC Matrix")
            with open(excel_filename, "rb") as file:
                st.download_button(
                    label="📥 Download SAC Matrix as Excel",
                    data=file,
                    file_name=excel_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        elif metric == "BIC-NL":
            st.subheader("Bit Independence Criterion - Nonlinearity (BIC-NL)")
            n = 8
            sboxes = [sbox] * 9
            bic_nl_values = bic_nl_matrix(sboxes, n)
            columns = [f"S-box{i+1}" for i in range(len(sboxes))]
            df_bic_nl = pd.DataFrame(bic_nl_values, index=columns, columns=columns)
            st.write("**BIC-NL Matrix:**")
            st.dataframe(df_bic_nl, use_container_width=True)

            excel_filename = "bic_nl_matrix.xlsx"
            df_bic_nl.to_excel(excel_filename, index=True, sheet_name="BIC-NL Matrix")
            with open(excel_filename, "rb") as file:
                st.download_button(
                    label="📥 Download BIC-NL Matrix as Excel",
                    data=file,
                    file_name=excel_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        elif metric == "BIC-SAC":
            st.subheader("Bit Independence Criterion - SAC (BIC-SAC)")
            bic_sac_matrix = generate_bic_sac_matrix(sbox)
            df_bic_sac = pd.DataFrame(bic_sac_matrix, index=[f"Bit {i}" for i in range(8)], columns=[f"Bit {j}" for j in range(8)])
            st.write("**BIC-SAC Matrix:**")
            st.dataframe(df_bic_sac, use_container_width=True)

            excel_filename = "bic_sac_matrix.xlsx"
            df_bic_sac.to_excel(excel_filename, index=True, sheet_name="BIC-SAC Matrix")
            with open(excel_filename, "rb") as file:
                st.download_button(
                    label="📥 Download BIC-SAC Matrix as Excel",
                    data=file,
                    file_name=excel_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        elif metric == "LAP":
            st.subheader("Linear Approximation Probability (LAP)")
            n = 8
            lap_value, result_data = lap(sbox, n)
            st.success(f"**LAP Value:** {lap_value}")
            df_lap = pd.DataFrame(result_data)

            excel_filename = "lap_results.xlsx"
            df_lap.to_excel(excel_filename, index=False)
            with open(excel_filename, "rb") as file:
                st.download_button(
                    label="📥 Download LAP Results as Excel",
                    data=file,
                    file_name=excel_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        elif metric == "DAP":
            st.subheader("Differential Approximation Probability (DAP)")
            n = 8
            dap_value, results = dap(sbox, n)
            st.success(f"**DAP Value:** {dap_value}")
            df_dap = pd.DataFrame(results)

            excel_filename = "dap_results.xlsx"
            df_dap.to_excel(excel_filename, index=False)
            with open(excel_filename, "rb") as file:
                st.download_button(
                    label="📥 Download DAP Results as Excel",
                    data=file,
                    file_name=excel_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
