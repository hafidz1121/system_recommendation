import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from dotenv import load_dotenv
from urllib.parse import quote_plus
import os
from collections import defaultdict

# Scikit-learn & Scikit-surprise
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import ndcg_score
from surprise import Reader, Dataset, KNNBasic
from surprise.model_selection import cross_validate, train_test_split as surprise_tts

print("--- Memulai Proses Evaluasi Lanjutan ---")

# --- Bagian 1: Koneksi ke Database ---
load_dotenv()
DB_USER = os.environ.get('DB_USERNAME')
DB_PASSWORD = os.environ.get('DB_PASSWORD')
DB_HOST = os.environ.get('DB_HOST')
DB_PORT = os.environ.get('DB_PORT')
DB_NAME = os.environ.get('DB_DATABASE')
encoded_password = quote_plus(DB_PASSWORD)
db_connection_str = f'mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
db_engine = create_engine(db_connection_str)

# --- Bagian 2: Pemuatan dan Pemrosesan Data ---
# Memuat semua data yang relevan: detail menu, rating, favorit, pembelian,
# dan interaksi lainnya, lalu menggabungkannya menjadi satu DataFrame utama.
print("\n[1/7] Memuat semua data interaksi dari database...")

menu_query = """
    SELECT
        menu.id, menu.name, menu.description, c.name AS canteen_name,
        GROUP_CONCAT(cat.name SEPARATOR ' ') AS categories
    FROM menu
    JOIN canteen c ON menu.canteen_id = c.id
    LEFT JOIN category_menu cm ON menu.id = cm.menu_id
    LEFT JOIN category cat ON cm.category_id = cat.id
    WHERE menu.deleted_at IS NULL
    GROUP BY menu.id, c.name
"""
menu_df = pd.read_sql(menu_query, db_engine)
# Membuat kolom 'main_category' untuk mempermudah evaluasi berbasis kategori.
menu_df['main_category'] = menu_df['categories'].fillna('Lainnya').apply(lambda x: x.split()[0])

# Memuat semua jenis interaksi dan memberinya bobot.
ratings_query = "SELECT customer_id, menu_id, rating FROM menu_rating WHERE deleted_at IS NULL"
ratings_data = pd.read_sql(ratings_query, db_engine)
if not ratings_data.empty:
    ratings_data['weight'] = ratings_data['rating'] / 5.0
    ratings_data['interaction_type'] = 'rating'
    ratings_data.drop(columns=['rating'], inplace=True)

favorites_query = "SELECT customer_id, menu_id FROM favorite_menu"
favorites_data = pd.read_sql(favorites_query, db_engine)
if not favorites_data.empty:
    favorites_data['weight'] = 1.0
    favorites_data['interaction_type'] = 'favorite'

purchase_query = "SELECT t.customer_id, td.menu_id FROM transaction_detail td JOIN transaction t ON td.transaction_id = t.id WHERE t.status = 'done' AND t.deleted_at IS NULL"
purchase_data = pd.read_sql(purchase_query, db_engine)
if not purchase_data.empty:
    purchase_data['weight'] = 0.9
    purchase_data['interaction_type'] = 'checkout'

implicit_query = "SELECT customer_id, menu_id, interaction_type FROM customer_interactions WHERE menu_id IS NOT NULL AND interaction_type IN ('view_detail_menu', 'add_to_cart')"
implicit_data = pd.read_sql(implicit_query, db_engine)
if not implicit_data.empty:
    implicit_data['weight'] = implicit_data['interaction_type'].map({'add_to_cart': 0.6, 'view_detail_menu': 0.4})
    # Hapus kolom interaction_type asli agar tidak duplikat
    implicit_data.drop(columns=['interaction_type'], inplace=True)

# Tambahkan kolom interaction_type ke DataFrame lain agar bisa digabung
if not ratings_data.empty: ratings_data['interaction_type'] = 'rating'
if not favorites_data.empty: favorites_data['interaction_type'] = 'favorite'
if not purchase_data.empty: purchase_data['interaction_type'] = 'checkout'

# Untuk implicit, kita buat kolomnya dari mapping
if not implicit_data.empty:
    implicit_data_copy = implicit_data.copy()
    implicit_data_copy['interaction_type'] = implicit_data_copy['weight'].map({0.6: 'add_to_cart', 0.4: 'view_detail_menu'})
    all_interactions = [ratings_data, favorites_data, purchase_data, implicit_data_copy]
else:
    all_interactions = [ratings_data, favorites_data, purchase_data]

valid_interactions = [df for df in all_interactions if not df.empty and 'weight' in df.columns]
if not valid_interactions:
    print("\nPROSES DIHENTIKAN: Tidak ada data interaksi valid.")
    exit()
interactions_df_with_types = pd.concat(valid_interactions, ignore_index=True)
interactions_df_with_types.dropna(subset=['customer_id', 'menu_id', 'weight'], inplace=True)

# Simpan data dengan tipe interaksi untuk analisis nanti
interactions_df = interactions_df_with_types.sort_values(by='weight', ascending=False).drop_duplicates(subset=['customer_id', 'menu_id'], keep='first')
print(f"Semua data interaksi berhasil dimuat. Total interaksi unik: {len(interactions_df)}")

# --- Bagian 3: Analisis Threshold & Persiapan Model ---
# Menganalisis distribusi interaksi untuk menentukan threshold cold-start
# dan mempersiapkan model TF-IDF (Content-Based) serta KNN (Collaborative).
# ========================================================================
print("\n[2/7] Menganalisis threshold...")
interaction_counts = interactions_df['customer_id'].value_counts()
stats = interaction_counts.describe()
q1 = stats.get('25%', 0)
RECOMMENDED_THRESHOLD = int(q1) + 1 if pd.notna(q1) else 5
print(f"  -> Threshold yang disarankan dari data: {RECOMMENDED_THRESHOLD}")

print("\n[3/7] Mempersiapkan model TF-IDF dan KNN...")

# Persiapan model Content-Based (TF-IDF)
menu_df['combined_features'] = menu_df['name'].fillna('') + ' ' + menu_df['categories'].fillna('') + ' ' + menu_df['canteen_name'].fillna('') + ' ' + menu_df['description'].fillna('')
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(menu_df['combined_features'])

# Persiapan model Collaborative Filtering (KNN)
reader = Reader(rating_scale=(0.1, 1.0))
data = Dataset.load_from_df(interactions_df[['customer_id', 'menu_id', 'weight']], reader)
trainset_full = data.build_full_trainset()
sim_options = {'name': 'cosine', 'user_based': True}
knn = KNNBasic(sim_options=sim_options, verbose=False)
knn.fit(trainset_full)

# --- Bagian 4: Evaluasi Akurasi Prediksi untuk CF ---
# Menghitung seberapa akurat model CF dalam memprediksi nilai 'weight' (rating).
# Metrik yang digunakan adalah MAE dan RMSE.
# ========================================================================
print("\n[4/7] Mengevaluasi akurasi Collaborative Filtering...")
cf_accuracy_results = cross_validate(knn, data, measures=['RMSE', 'MAE'], cv=3, verbose=False)
# --- Prediksi pada test set untuk visualisasi ---
trainset_pr, testset_pr = surprise_tts(data, test_size=.25, random_state=42)
knn.fit(trainset_pr)
cf_predictions = knn.test(testset_pr)

# --- Bagian 5: Fungsi Evaluasi Relevansi Lanjutan ---
# Mendefinisikan fungsi-fungsi baru untuk mengukur kualitas rekomendasi
# dengan metrik yang lebih baik: Precision, Recall, Category Precision, dan NDCG.
# ========================================================================
def calculate_all_metrics(recommendations, ground_truth, menu_df, top_n=10):
    """Menghitung semua metrik relevansi dan peringkat."""
    recs = recommendations[:top_n]
    
    # Metrik standar: Seberapa banyak tebakan item yang sama persis.
    hits = len(set(recs) & set(ground_truth))
    precision = hits / len(recs) if recs else 0
    recall = hits / len(ground_truth) if ground_truth else 0

    # Metrik baru: Seberapa banyak tebakan kategori yang cocok.
    rec_categories = set(menu_df.loc[menu_df['id'].isin(recs), 'main_category'])
    gt_categories = set(menu_df.loc[menu_df['id'].isin(ground_truth), 'main_category'])
    category_hits = len(rec_categories & gt_categories)
    category_precision = category_hits / len(rec_categories) if rec_categories else 0

    # Metrik baru: Mengukur kualitas urutan rekomendasi.
    true_relevance = np.asarray([1 if item in ground_truth else 0 for item in recs]).reshape(1, -1)
    ndcg = ndcg_score(true_relevance, np.arange(len(recs)).reshape(1, -1) + 1, k=top_n) if np.sum(true_relevance) > 0 else 0.0

    return {"precision": precision, "recall": recall, "category_precision": category_precision, "ndcg": ndcg}

def run_relevance_evaluation(train_df, test_df, menu_df, tfidf_matrix, knn_model, threshold, top_n=10):
    """Loop utama untuk menjalankan evaluasi pada semua pengguna di test set."""
    cb_results, hybrid_results = defaultdict(list), defaultdict(list)
    
    user_interaction_counts = train_df['customer_id'].value_counts()
    test_user_groups = test_df.groupby('customer_id')
    
    for user_id, test_group in test_user_groups:
        if user_id not in user_interaction_counts: continue
        
        ground_truth = test_group['menu_id'].tolist()
        interaction_count = user_interaction_counts.get(user_id, 0)
        
        # --- Evaluasi Content-Based ---
        trigger_menu_id = train_df[train_df['customer_id'] == user_id].iloc[0]['menu_id']
        try:
            idx = menu_df.index[menu_df['id'] == trigger_menu_id][0]
        except IndexError: continue
        cosine_sim = linear_kernel(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
        sim_scores_indices = np.argsort(cosine_sim)[::-1]
        cb_recs = [menu_df['id'].iloc[i] for i in sim_scores_indices if menu_df['id'].iloc[i] != trigger_menu_id][:top_n]
        cb_metrics = calculate_all_metrics(cb_recs, ground_truth, menu_df, top_n)
        for key, val in cb_metrics.items(): cb_results[key].append(val)
            
        # --- Evaluasi Hybrid ---
        if interaction_count < threshold:
            hybrid_recs = cb_recs
        else:
            rated_items = train_df[train_df['customer_id'] == user_id]['menu_id'].tolist()
            unrated_items = [item for item in menu_df['id'].unique() if item not in rated_items]
            predictions = [knn_model.predict(user_id, item_id) for item_id in unrated_items]
            predictions.sort(key=lambda x: x.est, reverse=True)
            hybrid_recs = [pred.iid for pred in predictions[:top_n]]
        
        hybrid_metrics = calculate_all_metrics(hybrid_recs, ground_truth, menu_df, top_n)
        for key, val in hybrid_metrics.items(): hybrid_results[key].append(val)

    # Menghitung rata-rata dari semua hasil
    avg_cb = {key: np.mean(val) for key, val in cb_results.items()}
    avg_hybrid = {key: np.mean(val) for key, val in hybrid_results.items()}
    return avg_cb, avg_hybrid

# --- Bagian 6: Menjalankan Proses Evaluasi Relevansi ---
# Membagi data menjadi data latih dan data uji, lalu memanggil fungsi evaluasi utama.
# ========================================================================
print("\n[5/7] Menjalankan evaluasi relevansi untuk CB dan Hybrid...")
train_set, test_set = train_test_split(interactions_df, test_size=0.2, random_state=42)
cb_eval_results, hybrid_eval_results = run_relevance_evaluation(train_set, test_set, menu_df, tfidf_matrix, knn, RECOMMENDED_THRESHOLD)
print("Evaluasi relevansi selesai.")

# --- Bagian 7: Menampilkan Hasil Akhir ---
# Mencetak semua hasil metrik ke terminal dengan format yang mudah dibaca.
# ========================================================================
print("\n[6/7] Hasil Akhir Evaluasi Lanjutan:")
print("="*60)
print("Metode Collaborative Filtering (KNN) - Metrik Akurasi")
print(f"  - Akurasi Prediksi (MAE) : {np.mean(cf_accuracy_results['test_mae']):.4f}")
print(f"  - Akurasi Prediksi (RMSE): {np.mean(cf_accuracy_results['test_rmse']):.4f}")
print("="*60)
print("Metode Content-Based Filtering (TF-IDF)")
print(f"  - Relevansi (Precision)             : {cb_eval_results.get('precision', 0):.4f}")
print(f"  - Relevansi (Recall)                : {cb_eval_results.get('recall', 0):.4f}")
print(f"  - Relevansi Kategori (Cat. Precision) : {cb_eval_results.get('category_precision', 0):.4f}  <-- (Perhatikan Skor Ini)")
print(f"  - Kualitas Peringkat (NDCG)         : {cb_eval_results.get('ndcg', 0):.4f}")
print("="*60)
print(f"Metode Hybrid Filtering (Threshold = {RECOMMENDED_THRESHOLD})")
print(f"  - Relevansi (Precision)             : {hybrid_eval_results.get('precision', 0):.4f}")
print(f"  - Relevansi (Recall)                : {hybrid_eval_results.get('recall', 0):.4f}")
print(f"  - Relevansi Kategori (Cat. Precision) : {hybrid_eval_results.get('category_precision', 0):.4f}  <-- (Perhatikan Skor Ini)")
print(f"  - Kualitas Peringkat (NDCG)         : {hybrid_eval_results.get('ndcg', 0):.4f}")
print("="*60)

# --- Bagian 8: Membuat dan Menyimpan Semua Grafik ---
print("\n[7/7] Membuat visualisasi hasil evaluasi...")

# --- PERUBAHAN 1: Buat folder jika belum ada ---
output_folder = "grafik_visualisasi"
os.makedirs(output_folder, exist_ok=True)

# Grafik 1: Distribusi Interaksi
plt.figure(figsize=(10, 6))
sns.histplot(interaction_counts, bins=30, kde=True)
plt.title('Distribusi Jumlah Interaksi per Pengguna', fontsize=16)
plt.xlabel('Jumlah Interaksi', fontsize=12)
plt.ylabel('Jumlah Pengguna', fontsize=12)
plt.tight_layout()
# --- PERUBAHAN 2: Simpan ke dalam folder ---
plt.savefig(os.path.join(output_folder, 'distribusi_interaksi.png'))
print(f"Grafik '{os.path.join(output_folder, 'distribusi_interaksi.png')}' telah disimpan.")

# Grafik 2: Popularitas Menu
menu_popularity = interactions_df['menu_id'].value_counts().head(15).reset_index()
menu_popularity.columns = ['menu_id', 'jumlah_interaksi']
menu_popularity = pd.merge(menu_popularity, menu_df[['id', 'name']], left_on='menu_id', right_on='id')
plt.figure(figsize=(12, 8))
sns.barplot(x='jumlah_interaksi', y='name', data=menu_popularity, palette='viridis')
plt.title('Top 15 Menu Paling Populer', fontsize=16)
plt.xlabel('Jumlah Interaksi', fontsize=12)
plt.ylabel('Nama Menu', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'popularitas_menu.png'))
print(f"Grafik '{os.path.join(output_folder, 'popularitas_menu.png')}' telah disimpan.")

# Grafik 3: Scatter Plot Akurasi Prediksi CF
predictions_df = pd.DataFrame(cf_predictions, columns=['uid', 'iid', 'true_rating', 'pred_rating', 'details'])
plt.figure(figsize=(8, 8))
sns.regplot(x='true_rating', y='pred_rating', data=predictions_df,
            scatter_kws={'alpha':0.3}, line_kws={'color': 'red'})
plt.title('Akurasi Prediksi CF (Nilai Aktual vs. Prediksi)', fontsize=14)
plt.xlabel('Nilai Weight Aktual', fontsize=12)
plt.ylabel('Nilai Weight Prediksi', fontsize=12)
plt.plot([0.1, 1.0], [0.1, 1.0], linestyle='--', color='green', label='Prediksi Sempurna')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'akurasi_prediksi_cf.png'))
print(f"Grafik '{os.path.join(output_folder, 'akurasi_prediksi_cf.png')}' telah disimpan.")

# Grafik 4: Bar Chart Error MAE & RMSE untuk CF
cf_metrics_df = pd.DataFrame({
    'Metric': ['MAE', 'RMSE'],
    'Score': [np.mean(cf_accuracy_results['test_mae']), np.mean(cf_accuracy_results['test_rmse'])]
})
plt.figure(figsize=(7, 5))
ax = sns.barplot(x='Metric', y='Score', data=cf_metrics_df, palette='coolwarm')
plt.title('Error Rata-rata Collaborative Filtering', fontsize=14)
plt.ylabel('Skor Error (lebih rendah lebih baik)', fontsize=12)
plt.ylim(0, max(cf_metrics_df['Score']) * 1.2)
for p in ax.patches:
    ax.annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 9), textcoords='offset points')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'error_cf.png'))
print(f"Grafik '{os.path.join(output_folder, 'error_cf.png')}' telah disimpan.")

# Grafik 5: Perbandingan Metrik Relevansi CBF & Hybrid
results_data = {
    'Model': ['Content-Based', 'Hybrid', 'Content-Based', 'Hybrid', 'Content-Based', 'Hybrid'],
    'Metric': ['Precision', 'Precision', 'Category Precision', 'Category Precision', 'NDCG', 'NDCG'],
    'Score': [
        cb_eval_results.get('precision', 0), hybrid_eval_results.get('precision', 0),
        cb_eval_results.get('category_precision', 0), hybrid_eval_results.get('category_precision', 0),
        cb_eval_results.get('ndcg', 0), hybrid_eval_results.get('ndcg', 0)
    ]
}
results_df = pd.DataFrame(results_data)
plt.figure(figsize=(12, 7))
ax = sns.barplot(x='Metric', y='Score', hue='Model', data=results_df, palette='pastel')
plt.title('Perbandingan Metrik Relevansi (CBF vs Hybrid)', fontsize=16)
plt.ylabel('Skor', fontsize=12)
plt.xlabel('Metrik', fontsize=12)
plt.ylim(0, 1.0)
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 9), textcoords='offset points')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'perbandingan_metrik.png'))
print(f"Grafik '{os.path.join(output_folder, 'perbandingan_metrik.png')}' telah disimpan.")

# --- Popularitas Kantin ---
# Gabungkan interaksi dengan menu untuk mendapatkan nama kantin
interactions_with_canteen = pd.merge(interactions_df_with_types, menu_df[['id', 'canteen_name']], left_on='menu_id', right_on='id')
canteen_popularity = interactions_with_canteen['canteen_name'].value_counts().head(10)

plt.figure(figsize=(12, 7))
sns.barplot(x=canteen_popularity.values, y=canteen_popularity.index, palette='rocket')
plt.title('Top 10 Kantin Terpopuler Berdasarkan Jumlah Interaksi', fontsize=16)
plt.xlabel('Jumlah Total Interaksi', fontsize=12)
plt.ylabel('Nama Kantin', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'popularitas_kantin.png'))
print(f"Grafik '{os.path.join(output_folder, 'popularitas_kantin.png')}' telah disimpan.")


# --- Distribusi Jenis Interaksi ---
interaction_type_counts = interactions_df['interaction_type'].value_counts()

plt.figure(figsize=(10, 8))
plt.pie(interaction_type_counts, labels=interaction_type_counts.index, autopct='%1.1f%%', startangle=140,
        colors=sns.color_palette('YlGnBu'))
plt.title('Distribusi Jenis Interaksi Pengguna', fontsize=16)
plt.ylabel('') # Hapus label y yang tidak perlu
plt.savefig(os.path.join(output_folder, 'distribusi_jenis_interaksi.png'))
print(f"Grafik '{os.path.join(output_folder, 'distribusi_jenis_interaksi.png')}' telah disimpan.")


print("\n--- Semua Proses Selesai ---")
