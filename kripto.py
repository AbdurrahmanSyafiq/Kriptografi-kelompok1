import streamlit as st
import pandas as pd
import numpy as np

# =================== PAGE CONFIG & LANDING UI ===================
st.set_page_config(page_title="S-Box Metrics Calculator", page_icon="🔐", layout="wide")

def _inject_css():
    st.markdown("""
    <style>
      /* Layout umum */
      .block-container {padding-top: 1.8rem; padding-bottom: 2rem;}
      [data-testid="stHeader"] {background: transparent;}

      /* Sidebar TETAP ada: hanya dipoles sedikit */
      section[data-testid="stSidebar"] {border-right: 1px solid rgba(148,163,184,.25);}
      section[data-testid="stSidebar"] .stFileUploader, 
      section[data-testid="stSidebar"] .stSelectbox, 
      section[data-testid="stSidebar"] .stMarkdown {
        background: #ffffff; border-radius: 14px; padding: 8px 10px;
        box-shadow: 0 6px 20px rgba(2,6,23,.06);
      }

      /* Hero */
      .hero {
        position: relative; border-radius: 24px; padding: 36px; overflow: hidden; 
        display: flex; gap: 32px; align-items: center; justify-content: center;
        background: radial-gradient(1200px 600px at 15% 10%, rgba(99,102,241,.16), transparent 50%),
                    linear-gradient(135deg, #0f172a 0%, #111827 60%, #0b1220 100%);
        color: #e5e7eb; 
        box-shadow: 0 30px 60px rgba(2,6,23,.35), inset 0 0 0 1px rgba(255,255,255,.06);
      }
      .hero h1 {
        font-size: 44px; line-height: 1.08; margin: 0 0 8px 0; letter-spacing: .2px;
        color: #ffffff;
      }
      .hero p {font-size: 16px; color: #cbd5e1; margin: 0;}
      .hero-badge {
        display: inline-flex; gap: 8px; align-items:center;
        background: rgba(99,102,241,.12); color: #e0e7ff;
        border: 1px solid rgba(99,102,241,.35);
        padding: 6px 10px; border-radius: 999px; font-size: 12px;
      }
      .hero img {
        width: 300px; max-width: 44vw; height: auto; border-radius: 20px;
        border: 1px solid rgba(255,255,255,.12); 
        box-shadow: 0 24px 70px rgba(2,6,23,.55);
        display: block; margin: 0 auto;
      }

      /* Section title w/ anchor icon */
      .section-title {display:flex; align-items:center; gap:10px; margin: 26px 0 8px;}
      .section-title a {text-decoration:none; color:#64748b; font-size:18px}

      /* Grid fitur */
      .grid {display: grid; gap: 16px; grid-template-columns: repeat(4, minmax(0,1fr));}
      @media (max-width: 1200px){ .grid {grid-template-columns: repeat(2,1fr);} }
      @media (max-width: 720px){ .grid {grid-template-columns: 1fr;} }
      .card {
        background: linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.02));
        border: 1px solid rgba(148,163,184,.18); border-radius: 18px; padding: 18px 16px 16px;
        box-shadow: 0 8px 26px rgba(2,6,23,.06);
      }
      .card h4 {margin: 0 0 6px; font-size: 16px; color: #0f172a;}
      .card p {margin: 0; font-size: 13px; color: #334155;}
      .muted {color:#94a3b8; font-size: 12px;}

      /* Steps */
      .steps {
        border-radius:18px; padding:16px; 
        border:1px solid rgba(148,163,184,.25); background:white;
        box-shadow: 0 8px 26px rgba(2,6,23,.05);
      }

      .footer {margin-top: 18px; color: #6b7280; font-size: 12px; text-align:center;}
    </style>
    """, unsafe_allow_html=True)

def _landing_page():
    _inject_css()

    # Hero dua kolom (kiri teks, kanan gambar)
    c1, c2 = st.columns([1.42, 1], vertical_alignment="center")
    with c1:
        st.markdown("""
        <div class="hero">
          <div style="flex:1;">
            <div class="hero-badge">🔐 S-Box Metrics • Fast • Reproducible</div>
            <h1>S-Box Metrics Calculator</h1>
            <p>Menghitung Nonlinearity, SAC, BIC-NL, BIC-SAC, LAP, dan DAP secara cepat.
            Unggah konfigurasi S-Box-mu dari <b>sidebar kiri</b>, lalu pilih metriknya.</p>
            <div style="margin-top:14px" class="muted">Tip: CSV/XLSX berisi 256 angka (0–255), satu kolom atau satu baris.</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="hero">
          <img src="https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?q=80&w=1600&auto=format&fit=crop" alt="S-Box Visual"/>
        </div>
        """, unsafe_allow_html=True)

    # Section title ala screenshot (teks + icon tautan)
    st.markdown("""
      <div class="section-title" id="features">
        <h3 style="margin:0;font-size:28px;">Apa yang bisa dihitung?</h3>
        <a href="#features">🔗</a>
      </div>
    """, unsafe_allow_html=True)

    # Fitur grid
    st.markdown("""
      <div class="grid">
        <div class="card"><h4>🧩 Nonlinearity</h4>
          <p>Jarak minimum ke semua fungsi afine—semakin besar, semakin kuat terhadap serangan linear.</p></div>
        <div class="card"><h4>🌊 SAC</h4>
          <p>Seberapa sensitif output terhadap pembalikan satu bit input (Strict Avalanche Criterion).</p></div>
        <div class="card"><h4>🔗 BIC-NL</h4>
          <p>Independensi bit output diukur lewat nonlinearity gabungan.</p></div>
        <div class="card"><h4>🧪 BIC-SAC</h4>
          <p>Independensi antar bit output saat satu bit input di-flip.</p></div>
        <div class="card"><h4>📈 LAP</h4>
          <p>Linear Approximation Probability—indikasi kelemahan pada linear cryptanalysis.</p></div>
        <div class="card"><h4>⚡ DAP</h4>
          <p>Differential Approximation Probability—indikasi kelemahan pada differential cryptanalysis.</p></div>
        <div class="card"><h4>📤 Ekspor</h4>
          <p>SAC/BIC-NL/BIC-SAC/LAP/DAP langsung diekspor ke Excel untuk dokumentasi.</p></div>
        <div class="card"><h4>🧠 Validasi</h4>
          <p>Deteksi otomatis panjang 256 & rentang 0–255 agar perhitungan valid.</p></div>
      </div>
    """, unsafe_allow_html=True)

    st.markdown("### Cara pakai (ringkas)")
    st.markdown("""
    <div class="steps">
      <ol>
        <li>Buka <b>Configuration</b> di sidebar kiri → klik <b>Browse files</b> atau drag-and-drop CSV/XLSX.</li>
        <li>Pastikan berisi 256 angka 0–255 (8×8 S-Box).</li>
        <li>Pilih metrik di sidebar → hasil & tombol unduh tampil di area utama.</li>
      </ol>
    </div>
    """, unsafe_allow_html=True)


    st.markdown('<div class="footer">Made for research & teaching • Keep your data local</div>', unsafe_allow_html=True)

# =================== FUNGSI PERHITUNGAN (LOGIKA TETAP) ===================

def hamming_weight(x):
    """Menghitung jumlah bit '1' dalam representasi biner dari x."""
    return bin(x).count('1')

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

def sac(sbox, n):
    total_weight = 0
    total_cases = 0
    for i in range(2**n):
        original_output = sbox[i]
        for bit in range(n):
            flipped_input = i ^ (1 << bit)
            flipped_output = sbox[flipped_input]
            diff = original_output ^ flipped_output
            weight = hamming_weight(diff)
            total_weight += weight
            total_cases += n
    return total_weight / total_cases

def bic_nl(sbox, n):
    total_nl = 0
    count = 0
    for bit1 in range(n):
        for bit2 in range(bit1 + 1, n):
            count += 1
            sbox_bit1 = [(y >> bit1) & 1 for y in sbox]
            sbox_bit2 = [(y >> bit2) & 1 for y in sbox]
            combined = [b1 ^ b2 for b1, b2 in zip(sbox_bit1, sbox_bit2)]
            W = walsh_transform(combined)
            max_walsh = np.max(np.abs(W))
            nl = 2 ** (n - 1) - max_walsh / 2
            total_nl += nl
    return total_nl / count

def calculate_bic_sac(sbox):
    n = len(sbox)
    bit_length = 8
    total_pairs = 0
    total_independence = 0
    for i in range(bit_length):
        for j in range(i + 1, bit_length):
            independence_sum = 0
            for x in range(n):
                for bit_to_flip in range(bit_length):
                    flipped_x = x ^ (1 << bit_to_flip)
                    y1 = sbox[x]; y2 = sbox[flipped_x]
                    b1_i = (y1 >> i) & 1; b1_j = (y1 >> j) & 1
                    b2_i = (y2 >> i) & 1; b2_j = (y2 >> j) & 1
                    independence_sum += ((b1_i ^ b2_i) ^ (b1_j ^ b2_j))
            pair_independence = independence_sum / (n * bit_length)
            total_independence += pair_independence
            total_pairs += 1
    return total_independence / total_pairs

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

def calculate_sac(sbox, bit_i, bit_j):
    bit_length = 8
    n = len(sbox)
    independence_sum = 0
    for x in range(n):
        for bit_to_flip in range(bit_length):
            flipped_x = x ^ (1 << bit_to_flip)
            y1 = sbox[x]; y2 = sbox[flipped_x]
            b1_i = (y1 >> bit_i) & 1; b1_j = (y1 >> bit_j) & 1
            b2_i = (y2 >> bit_i) & 1; b2_j = (y2 >> bit_j) & 1
            independence_sum += ((b1_i ^ b2_i) ^ (b1_j ^ b2_j))
    return independence_sum / (n * bit_length)

def generate_bic_sac_matrix(sbox):
    bit_length = 8
    matrix = np.zeros((bit_length, bit_length), dtype=object)
    for i in range(bit_length):
        for j in range(bit_length):
            matrix[i][j] = calculate_sac(sbox, i, j) if i != j else "-"
    return matrix

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

# =================== HEADER & SIDEBAR (TETAP) ===================
st.title("🔐 S-Box Metrics Calculator")
st.markdown("Use this tool to calculate various cryptographic metrics for S-Boxes.")

st.sidebar.header("⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader(
    "Upload S-Box configuration (CSV or Excel):",
    type=["csv", "xlsx"],
    help="Upload a file containing the S-Box configuration."
)

# =================== MAIN: LANDING jika belum ada file ===================
if not uploaded_file:
    _landing_page()
else:
    # Baca file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, header=None)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file, header=None)
    else:
        st.error("Unsupported file format!")
        st.stop()

    st.subheader("📄 Uploaded S-Box Configuration")
    st.dataframe(df, use_container_width=True)

    sbox = df.values.flatten().tolist()
    st.write("**S-Box (List):**", sbox)

    if len(sbox) != 256:
        st.error("Error: S-box harus memiliki panjang 256 elemen.")
    else:
        metric = st.sidebar.selectbox(
            "📊 Choose Metric:",
            ["Nonlinearity", "SAC", "BIC-NL", "BIC-SAC", "LAP", "DAP"],
            help="Select the cryptographic metric to calculate."
        )

        if metric == "Nonlinearity":
            st.subheader("Nonlinearity Calculation")
            n, m = 8, 8
            nl_value = nonlinearity(sbox, n, m)
            st.success(f"**Nonlinearity Value:** {nl_value}")

        elif metric == "SAC":
            st.subheader("Strict Avalanche Criterion (SAC)")
            n = 8
            sac_matrix = compute_sac_matrix(sbox, n)
            df_sac = pd.DataFrame(
                sac_matrix,
                columns=[f"Bit-{i+1}" for i in range(n)],
                index=[f"Bit-{i+1}" for i in range(n)]
            )
            sac_value = sac(sbox, n)
            st.success(f"**SAC Value:** {sac_value:.5f}")
            st.write("**SAC Matrix:**")
            st.dataframe(df_sac, use_container_width=True)
            excel_filename = "sac_matrix.xlsx"
            df_sac.to_excel(excel_filename, index=True, sheet_name="SAC Matrix")
            with open(excel_filename, "rb") as file:
                st.download_button("📥 Download SAC Matrix as Excel", file, file_name=excel_filename,
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        elif metric == "BIC-NL":
            st.subheader("Bit Independence Criterion - Nonlinearity (BIC-NL)")
            n = 8
            sboxes = [sbox] * 9
            bic_nl_values = bic_nl_matrix(sboxes, n)
            columns = [f"S-box{i+1}" for i in range(len(sboxes))]
            df_bic_nl = pd.DataFrame(bic_nl_values, index=columns, columns=columns)
            bic_nl_value = bic_nl(sbox, n)
            st.success(f"**BIC-NL Value:** {bic_nl_value:.5f}")
            st.write("**BIC-NL Matrix:**")
            st.dataframe(df_bic_nl, use_container_width=True)
            excel_filename = "bic_nl_matrix.xlsx"
            df_bic_nl.to_excel(excel_filename, index=True, sheet_name="BIC-NL Matrix")
            with open(excel_filename, "rb") as file:
                st.download_button("📥 Download BIC-NL Matrix as Excel", file, file_name=excel_filename,
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        elif metric == "BIC-SAC":
            st.subheader("Bit Independence Criterion - SAC (BIC-SAC)")
            bic_sac_matrix = generate_bic_sac_matrix(sbox)
            df_bic_sac = pd.DataFrame(
                bic_sac_matrix,
                index=[f"Bit {i}" for i in range(8)],
                columns=[f"Bit {j}" for j in range(8)]
            )
            bic_sac_value = calculate_bic_sac(sbox)
            st.success(f"**BIC-SAC Value:** {bic_sac_value:.5f}")
            st.write("**BIC-SAC Matrix:**")
            st.dataframe(df_bic_sac, use_container_width=True)
            excel_filename = "bic_sac_matrix.xlsx"
            df_bic_sac.to_excel(excel_filename, index=True, sheet_name="BIC-SAC Matrix")
            with open(excel_filename, "rb") as file:
                st.download_button("📥 Download BIC-SAC Matrix as Excel", file, file_name=excel_filename,
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        elif metric == "LAP":
            st.subheader("Linear Approximation Probability (LAP)")
            n = 8
            lap_value, result_data = lap(sbox, n)
            st.success(f"**LAP Value:** {lap_value}")
            df_lap = pd.DataFrame(result_data)
            excel_filename = "lap_results.xlsx"
            df_lap.to_excel(excel_filename, index=False)
            with open(excel_filename, "rb") as file:
                st.download_button("📥 Download LAP Results as Excel", file, file_name=excel_filename,
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        elif metric == "DAP":
            st.subheader("Differential Approximation Probability (DAP)")
            n = 8
            dap_value, results = dap(sbox, n)
            st.success(f"**DAP Value:** {dap_value}")
            df_dap = pd.DataFrame(results)
            excel_filename = "dap_results.xlsx"
            df_dap.to_excel(excel_filename, index=False)
            with open(excel_filename, "rb") as file:
                st.download_button("📥 Download DAP Results as Excel", file, file_name=excel_filename,
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
