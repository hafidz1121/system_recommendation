from flask import Flask, request, jsonify
import pandas as pd
import os
from recommender import HybridRecommender
from sqlalchemy import create_engine
from dotenv import load_dotenv
from urllib.parse import quote_plus 

# Muat environment variables dari file .env
load_dotenv()

app = Flask(__name__)

# --- Koneksi Database ---
DB_USER = os.environ.get('DB_USERNAME')
DB_PASSWORD = os.environ.get('DB_PASSWORD')
DB_HOST = os.environ.get('DB_HOST')
DB_PORT = os.environ.get('DB_PORT')
DB_NAME = os.environ.get('DB_DATABASE')

if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
    print("FATAL ERROR: Pastikan semua variabel (DB_HOST, DB_PORT, dll.) ada di file .env Anda.")
    exit()

# Encode password untuk menangani karakter spesial seperti '@'
encoded_password = quote_plus(DB_PASSWORD)
db_connection_str = f'mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

db_engine = create_engine(db_connection_str)

# --- Load Data dan Inisialisasi Recommender ---
# File: app.py (Ganti seluruh bagian ini)

# --- Load Data dan Inisialisasi Recommender ---
print("Loading data from database...")
try:
    # 1. Muat data menu (untuk Content-Based dan pencarian detail)
    menu_query = """
        SELECT
            menu.id,
            menu.name,
            menu.description,
            c.name AS canteen_name,
            GROUP_CONCAT(cat.name SEPARATOR ' ') AS categories
        FROM
            menu
        JOIN
            canteen c ON menu.canteen_id = c.id
        LEFT JOIN
            category_menu cm ON menu.id = cm.menu_id
        LEFT JOIN
            category cat ON cm.category_id = cat.id
        WHERE
            menu.deleted_at IS NULL
        GROUP BY
            menu.id, c.name
    """
    menu_df = pd.read_sql(menu_query, db_engine)
    
    # Persiapan untuk model Content-Based
    menu_df['description'] = menu_df['description'].fillna('')
    
    # --- Pengumpulan Data Interaksi untuk Collaborative Filtering ---

    # 2. Muat Data Eksplisit: Rating dari tabel menu_rating
    print("Loading ratings data...")
    ratings_query = "SELECT customer_id, menu_id, rating FROM menu_rating WHERE deleted_at IS NULL"
    ratings_data = pd.read_sql(ratings_query, db_engine)
    if not ratings_data.empty:
        ratings_data['weight'] = ratings_data['rating'] / 5.0
        ratings_data['interaction_type'] = 'rating'
        ratings_data = ratings_data.drop(columns=['rating'])

    # 3. Muat Data Eksplisit: Favorit dari tabel favorite_menu
    print("Loading favorites data...")
    favorites_query = "SELECT customer_id, menu_id FROM favorite_menu"
    favorites_data = pd.read_sql(favorites_query, db_engine)
    if not favorites_data.empty:
        favorites_data['weight'] = 1.0
        favorites_data['interaction_type'] = 'favorite'
    
    # 4. Muat Data Transaksi dari tabel transaction
    print("Loading purchase (checkout) data...")
    purchase_query = """
        SELECT t.customer_id, td.menu_id
        FROM transaction_detail td
        JOIN transaction t ON td.transaction_id = t.id
        WHERE t.status = 'done' AND t.deleted_at IS NULL
    """
    purchase_data = pd.read_sql(purchase_query, db_engine)
    if not purchase_data.empty:
        purchase_data['weight'] = 0.9 
        purchase_data['interaction_type'] = 'checkout'

    # 5. Muat Data Implisit Lainnya (Views & Add to Cart)
    print("Loading other implicit interactions (views, add_to_cart)...")
    implicit_query = """
        SELECT customer_id, menu_id, interaction_type 
        FROM customer_interactions 
        WHERE menu_id IS NOT NULL AND interaction_type IN ('view_detail_menu', 'add_to_cart')
    """
    implicit_data = pd.read_sql(implicit_query, db_engine)
    if not implicit_data.empty:
        interaction_weights = {'add_to_cart': 0.6, 'view_detail_menu': 0.4}
        implicit_data['weight'] = implicit_data['interaction_type'].map(interaction_weights)

    # 6. Gabungkan SEMUA Data Interaksi menjadi satu DataFrame
    print("Combining all interaction data...")
    all_interactions = [
        ratings_data, 
        favorites_data, 
        purchase_data, 
        implicit_data
    ]
    
    valid_interactions = [df for df in all_interactions if not df.empty]
    
    if not valid_interactions:
        raise ValueError("Tidak ada data interaksi yang bisa dimuat. Periksa isi tabel Anda.")

    interactions_df = pd.concat(valid_interactions, ignore_index=True)
    
    # 7. Hapus Duplikat & Prioritaskan Bobot Tertinggi
    interactions_df = interactions_df.sort_values(by='weight', ascending=False)
    interactions_df = interactions_df.drop_duplicates(subset=['customer_id', 'menu_id'], keep='first')

    print("Data loaded. Initializing recommender...")
    recommender = HybridRecommender(menu_df, interactions_df)
    print("Recommender ready.")

except Exception as e:
    print(f"DATABASE CONNECTION/DATA LOADING FAILED: {e}")
    exit()

# --- DEFINISI ENDPOINT /recommend ---
@app.route('/recommend/hybrid', methods=['POST'])
def get_hybrid_recommendations():
    """Endpoint ini menjalankan logika switching sesuai kondisi pengguna."""
    data = request.get_json()
    customer_id = data.get('customer_id')
    last_interacted_menu_id = data.get('last_interacted_menu_id')

    if not customer_id or not last_interacted_menu_id:
        return jsonify({'error': 'customer_id and last_interacted_menu_id are required'}), 400

    try:
        # Memanggil fungsi recommend() utama yang memiliki logika switching
        recommended_ids = recommender.recommend(customer_id, last_interacted_menu_id)
        if not recommended_ids:
            return jsonify([])
        recommended_menus = menu_df[menu_df['id'].isin(recommended_ids)].to_dict('records')
        return jsonify(recommended_menus)
    except Exception as e:
        print(f"ERROR during hybrid recommendation: {e}")
        return jsonify({'error': 'An internal error occurred.'}), 500


# 2. Endpoint HANYA untuk Metode CONTENT-BASED
@app.route('/recommend/content-based', methods=['POST'])
def get_content_based_recommendations():
    """Endpoint ini HANYA menjalankan Content-Based, mengabaikan riwayat pengguna."""
    data = request.get_json()
    last_interacted_menu_id = data.get('last_interacted_menu_id')

    if not last_interacted_menu_id:
        return jsonify({'error': 'last_interacted_menu_id is required'}), 400

    try:
        # Langsung memanggil fungsi content_based_recommendations()
        recommended_ids = recommender.content_based_recommendations(last_interacted_menu_id)
        if not recommended_ids:
            return jsonify([])
        recommended_menus = menu_df[menu_df['id'].isin(recommended_ids)].to_dict('records')
        return jsonify(recommended_menus)
    except Exception as e:
        print(f"ERROR during content-based recommendation: {e}")
        return jsonify({'error': 'An internal error occurred.'}), 500


# 3. Endpoint HANYA untuk Metode COLLABORATIVE FILTERING
@app.route('/recommend/collaborative', methods=['POST'])
def get_collaborative_recommendations():
    """Endpoint ini HANYA menjalankan Collaborative Filtering."""
    data = request.get_json()
    customer_id = data.get('customer_id')

    if not customer_id:
        return jsonify({'error': 'customer_id is required'}), 400

    try:
        # Langsung memanggil fungsi collaborative_filtering_recommendations()
        recommended_ids = recommender.collaborative_filtering_recommendations(customer_id)
        if not recommended_ids:
            return jsonify([])
        recommended_menus = menu_df[menu_df['id'].isin(recommended_ids)].to_dict('records')
        return jsonify(recommended_menus)
    except Exception as e:
        print(f"ERROR during collaborative recommendation: {e}")
        return jsonify({'error': 'An internal error occurred.'}), 500


# Contoh: app.run(host='192.168.1.37', port=5001, debug=True)
if __name__ == '__main__':
    app.run(host='192.168.1.37', port=5001, debug=True)