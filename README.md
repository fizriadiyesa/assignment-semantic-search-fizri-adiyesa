# Prototype Semantic Search & LLM untuk Customer Service E-commerce

## 1. Problem Statement dan Knowledge Base

Tim Customer Service e-commerce menerima banyak pertanyaan berulang terkait pemesanan, pengiriman, pembayaran, dan pengembalian barang. Sistem jawab secara manual dapat memperlambat respon dan membebankan operasional. Asisten CS ini berbasis semantic search dan Large Language Model agar jawaban dapat disusun berdasarkan FAQ yang tersedia, termasuk ketika pertanyaan pelanggan memakai parafrase.

Knowledge base bersumber dari `ecommerce_faq_knowledge_base.csv`, yang pada eksekusi notebook berisi 59 entri FAQ. Setiap entri memiliki kolom `faq_id`, `category`, `question`, dan `answer`. Satu unit dokumen yang dapat dicari (retrievable unit) merupakan gabungan `Q: {question} A: {answer}`. Knowledge base menjadi sumber informasi untuk jawaban; sistem tidak dimaksudkan mengarang kebijakan yang tidak tercantum.

## 2. Alur Sistem: Ingest → Embed → Store → Retrieve → Generate

CSV FAQ (59 baris)
  │ ingest: pandas membaca dan menggabungkan question + answer
  ▼
Dokumen FAQ
  │ embed: Gemini Embeddings API (gemini-embedding-001)
  ▼
Vektor float32 (59 × 3072)
  │ store: FAISS IndexFlatL2 (in-memory)
  ▼
Indeks FAISS (59 vektor)
  │ retrieve: query di-embed, pencarian L2, top_k=3
  ▼
3 FAQ terdekat
  │ generate: prompt Persona–Task–Context–Format + FAQ terambil
  ▼
Gemini LLM (gemini-3.5-flash, temperature=0.3)
  ▼
Jawaban CS + daftar sumber FAQ


1. Ingest: `pd.read_csv('../data/ecommerce_faq_knowledge_base.csv')` memuat FAQ. Pertanyaan dan jawaban digabung menjadi `content` agar keduanya turut menentukan kemiripan semantik.
2. Embed: `client.models.embed_content(model='gemini-embedding-001', contents=text)` menghasilkan embedding tiap dokumen. Hasil notebook menunjukkan matriks `(59, 3072)` bertipe `float32`.
3. Store: `faiss.IndexFlatL2(3072)` menyimpan 59 vektor untuk pencarian berdasarkan jarak Euclidean kuadrat (L2). Indeks ini dibuat di memori setiap notebook dijalankan, bukan disimpan secara persisten ke disk.
4. Retrieve: `semantic_search(query, top_k=3)` meng-embed pertanyaan pelanggan, mengambil tiga FAQ dengan jarak L2 terkecil, lalu mengembalikan `rank`, `question`, `answer`, `category`, dan `distance`. Distance lebih kecil berarti lebih dekat, bukan skor probabilitas atau persentase relevansi.
5. Generate: `build_prompt()` menginstruksikan model untuk menjawab hanya dari konteks FAQ dan menyarankan CS jika informasinya tidak tersedia. `generate_answer()` memanggil `gemini-3.5-flash` dengan `temperature=0.3`, lalu mengembalikan teks jawaban beserta sumber hasil retrieval.

## 3. Hasil Pengujian Query dan Analisis Relevansi

Dari lima query pada loop evaluasi, tiga menghasilkan jawaban substantif yang sesuai topik FAQ, sedangkan dua menghasilkan respons keterbatasan informasi. Untuk query yang tidak berhasil dijawab, LLM tidak menemukan FAQ yang relevan, sehingga memberikan jawaban bahwa informasi tidak tersedia. Hal ini menunjukkan bahwa kualitas dan cakupan knowledge base sangat penting untuk performa sistem.



## 4. Keterbatasan Sistem

- Cakupan FAQ terbatas: 59 entri belum menjawab semua kebijakan, misalnya biaya pengiriman cepat dan garansi produk elektronik.
- Retrieval selalu mengembalikan `top_k` hasil: Tidak ada ambang batas jarak untuk menolak dokumen yang kurang relevan. Contohnya, pertanyaan garansi memperoleh FAQ pengembalian sebagai hasil pertama.
- Risiko ketidakcocokan konteks: FAQ yang berdekatan secara semantik belum tentu menjawab rincian kondisi yang ditanyakan (misalnya pembatalan setelah pembayaran).
- Grounding hanya melalui prompt: Instruksi untuk tidak menjawab di luar FAQ mengurangi risiko jawaban tak berdasar, tetapi belum menjamin ketiadaan halusinasi. Tidak ada pemeriksaan otomatis yang memvalidasi setiap klaim jawaban terhadap sumber.



## 5. Cara Menjalankan Notebook

### Prasyarat

- Python 3 dan Jupyter Notebook/JupyterLab atau VS Code dengan ekstensi Jupyter.
- Paket: `pandas`, `numpy`, `faiss-cpu`, `google-genai`, `ipykernel` (serta `jupyterlab` bila menggunakan JupyterLab).
- Gemini API key yang memiliki akses ke model embedding dan generasi yang dipakai notebook.
- Dataset CSV pada lokasi yang sesuai dengan path relatif notebook.

### Struktur folder yang diharapkan

Notebook menggunakan path `../data/ecommerce_faq_knowledge_base.csv`, sehingga salah satu struktur yang sesuai adalah:

```text
project/
├── data/
│   └── ecommerce_faq_knowledge_base.csv
├── notebooks/
│   └── starter_notebook_semantic_search.ipynb
└── README.md
```

> **Penting:** Jika notebook ditempatkan di folder berbeda, ubah path pada `pd.read_csv(...)` sesuai lokasi CSV. File CSV tidak disertakan dalam dua file yang digunakan untuk menyusun README ini, sehingga harus disediakan secara terpisah.

### Instalasi dan eksekusi

Dari direktori `project/`:

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
python -m pip install pandas numpy faiss-cpu google-genai ipykernel jupyterlab
export GEMINI_API_KEY='YOUR_GEMINI_API_KEY'  # macOS/Linux
jupyter lab
```

Pada Windows PowerShell, aktivasi virtual environment dengan `.venv\Scripts\Activate.ps1` dan set API key menggunakan `$env:GEMINI_API_KEY='YOUR_GEMINI_API_KEY'`.

Buka `notebooks/starter_notebook_semantic_search.ipynb`, pilih kernel `.venv`, kemudian jalankan semua cell secara berurutan. Notebook akan membaca 59 FAQ, membentuk embedding, membuat indeks FAISS, menguji semantic search, dan memanggil LLM untuk menghasilkan jawaban.

**Catatan model:** Nama model yang tercatat di notebook adalah `gemini-embedding-001` dan `gemini-3.5-flash`. Ketersediaan model bergantung pada akses API dan dapat berubah; jika terjadi `404 NOT_FOUND`, periksa daftar model yang tersedia pada akun/API yang digunakan dan sesuaikan nama model di notebook. Jangan menaruh API key secara langsung di kode atau mengunggahnya ke repository.

### Contoh penggunaan fungsi di notebook

```python
results = semantic_search('kenapa paket saya belum sampai?', top_k=3)
for item in results:
    print(item['rank'], item['question'], item['distance'])

answer, sources = generate_answer('apakah bisa bayar pakai cicilan?')
print(answer)
```

Seluruh hasil pengujian dalam README ini merujuk pada output yang tersimpan dalam `starter_notebook_semantic_search.ipynb`; hasil dapat berbeda saat model, data, atau konfigurasi berubah.
