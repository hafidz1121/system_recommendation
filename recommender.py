import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Reader, Dataset, KNNBasic

class HybridRecommender:
    def __init__(self, menu_df, interactions_df):
        self.menu_df = menu_df
        self.interactions_df = interactions_df
        self.item_ids = self.menu_df['id'].unique().tolist()

        # --- Setup untuk Content-Based Filtering ---
        # 1. Isi nilai kosong (jika ada) di kolom-kolom fitur
        self.menu_df['categories'] = self.menu_df['categories'].fillna('')
        self.menu_df['description'] = self.menu_df['description'].fillna('')
        self.menu_df['canteen_name'] = self.menu_df['canteen_name'].fillna('')

        # 2. Gabungkan SEMUA fitur teks menjadi satu kolom 'combined_features'
        # Bobot bisa diatur dengan mengulang nama kolom, misal (self.menu_df['name'] * 2)
        self.menu_df['combined_features'] = (
            self.menu_df['name'] + ' ' +
            self.menu_df['categories'] + ' ' +
            self.menu_df['canteen_name'] + ' ' +
            self.menu_df['description']
        )

        # 3. Buat matriks TF-IDF dari fitur gabungan yang sudah kaya
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.menu_df['combined_features'])

        # --- Setup untuk Collaborative Filtering ---
        # Menggunakan interaksi 'rating' dan 'favorite'
        # Bobot untuk interaksi: rating > favorite
        interaction_weights = {'rating': 1.0, 'favorite': 0.8}
        
        # Cek apakah 'weight' sudah ada dari proses loading data
        if 'weight' not in interactions_df.columns:
            interactions_df['weight'] = interactions_df['interaction_type'].map(interaction_weights)

        reader = Reader(rating_scale=(0.1, 1.0)) # Skala berdasarkan bobot
        self.data = Dataset.load_from_df(interactions_df[['customer_id', 'menu_id', 'weight']], reader)
        self.trainset = self.data.build_full_trainset()
        
        # Konfigurasi KNN untuk menggunakan Cosine Similarity
        sim_options = {
            'name': 'cosine',
            'user_based': True  # True untuk user-based, False untuk item-based
        }
        # Menggunakan KNNBasic sebagai implementasi KNN
        self.knn = KNNBasic(sim_options=sim_options, verbose=False)
        self.knn.fit(self.trainset)

    # --- Metode diversify By Categori untuk perankingan rekomendasi ---
    def _diversify_by_category(self, recommended_ids, n=10):
        """Menyusun ulang daftar rekomendasi untuk memaksimalkan variasi kategori."""
        if not recommended_ids:
            return []

        # Ambil detail menu dari ID yang direkomendasikan
        recs_df = self.menu_df[self.menu_df['id'].isin(recommended_ids)].copy()
        
        # Ambil kategori utama (kata pertama dari string 'categories')
        # Ini untuk menyederhanakan, jika sebuah menu punya kategori "Makanan Snack", kita anggap "Makanan"
        recs_df['main_category'] = recs_df['categories'].apply(lambda x: x.split()[0] if x else 'Lainnya')

        # Urutkan DataFrame sesuai urutan skor rekomendasi awal
        recs_df = recs_df.set_index('id').loc[recommended_ids].reset_index()

        final_recs = []
        seen_categories = set()

        # Langkah 1: Ambil item teratas dari setiap kategori unik
        for _, row in recs_df.iterrows():
            if len(final_recs) >= n:
                break
            if row['main_category'] not in seen_categories:
                final_recs.append(row['id'])
                seen_categories.add(row['main_category'])

        # Langkah 2: Jika belum cukup, isi sisanya dengan item lain yang belum terpilih
        remaining_recs = [rec_id for rec_id in recommended_ids if rec_id not in final_recs]
        final_recs.extend(remaining_recs)

        return final_recs[:n]

    def content_based_recommendations(self, menu_id, n=10):
        try:
            # Mengambil index dari menu_id yang diberikan
            idx = self.menu_df[self.menu_df['id'] == menu_id].index[0]
        except IndexError:
            return [] # Return list kosong jika menu_id tidak ditemukan

        # Menghitung cosine similarity
        cosine_similarities = linear_kernel(self.tfidf_matrix[idx:idx+1], self.tfidf_matrix).flatten()
        # Mengambil index item yang paling mirip
        related_docs_indices = cosine_similarities.argsort()[:-n-2:-1]

        recommendations = []
        for i in related_docs_indices:
            # Jangan merekomendasikan item itu sendiri
            if self.menu_df['id'].iloc[i] != menu_id:
                recommendations.append(self.menu_df['id'].iloc[i])
                
        # --- Panggil fungsi diversifikasi sebelum return ---
        initial_recs = recommendations[:n]
        diversified_recs = self._diversify_by_category(initial_recs, n)
        return diversified_recs

    def collaborative_filtering_recommendations(self, customer_id, n=10):
        # Dapatkan daftar menu yang sudah berinteraksi dengan user
        rated_items = self.interactions_df[self.interactions_df['customer_id'] == customer_id]['menu_id'].unique().tolist()
        
        # Dapatkan semua menu yang BELUM berinteraksi
        unrated_items = [item_id for item_id in self.item_ids if item_id not in rated_items]

        # Lakukan prediksi untuk item yang belum di-rate menggunakan model KNN
        predictions = [self.knn.predict(customer_id, item_id) for item_id in unrated_items]
        
        # Urutkan prediksi berdasarkan estimasi rating tertinggi
        predictions.sort(key=lambda x: x.est, reverse=True)

        # 1. Ambil kandidat awal lebih banyak dari n (misal: 2*n) untuk memberikan
        #    bahan yang cukup bagi fungsi diversifikasi.
        num_candidates = n * 2 
        initial_recs = [pred.iid for pred in predictions[:num_candidates]]

        # 2. Panggil fungsi diversifikasi yang sama seperti pada Content-Based
        diversified_recs = self._diversify_by_category(initial_recs, n)
        
        return diversified_recs

    def recommend(self, customer_id, last_interacted_menu_id, n=10):
        # Ambil threshold, bisa diuji antara 3-11. Kita pakai 5 sebagai default.
        cold_start_threshold = 5 
        
        # Hitung jumlah interaksi unik (rating/favorite) dari pengguna
        user_interaction_count = self.interactions_df[
            (self.interactions_df['customer_id'] == customer_id) &
            (self.interactions_df['interaction_type'].isin(['rating', 'favorite']))
        ].shape[0]

        # Logika Switching
        if user_interaction_count < cold_start_threshold:
            print(f"INFO: User {customer_id} (interaksi: {user_interaction_count}) -> COLD START. Menggunakan Content-Based.")
            return self.content_based_recommendations(last_interacted_menu_id, n)
        else:
            print(f"INFO: User {customer_id} (interaksi: {user_interaction_count}) -> PENGGUNA LAMA. Menggunakan Collaborative Filtering.")
            return self.collaborative_filtering_recommendations(customer_id, n)