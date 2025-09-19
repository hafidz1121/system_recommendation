import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from urllib.parse import quote_plus
import os

print("--- Memulai Analisis Persona Pengguna (berdasarkan Email) ---")

# --- 1. KONEKSI KE DATABASE ---
load_dotenv()
DB_USER = os.environ.get('DB_USERNAME')
DB_PASSWORD = os.environ.get('DB_PASSWORD')
DB_HOST = os.environ.get('DB_HOST')
DB_PORT = os.environ.get('DB_PORT')
DB_NAME = os.environ.get('DB_DATABASE')
encoded_password = quote_plus(DB_PASSWORD)
db_connection_str = f'mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
db_engine = create_engine(db_connection_str)

# --- 2. AMBIL SEMUA DATA INTERAKSI & MASTER ---
print("Mengambil data interaksi, menu, dan customer...")
try:
    # 1. Muat Rating
    ratings_query = "SELECT customer_id, menu_id FROM menu_rating WHERE deleted_at IS NULL"
    ratings_data = pd.read_sql(ratings_query, db_engine)

    # 2. Muat Favorit
    favorites_query = "SELECT customer_id, menu_id FROM favorite_menu"
    favorites_data = pd.read_sql(favorites_query, db_engine)

    # 3. Muat Pembelian (Checkout)
    purchase_query = """
        SELECT t.customer_id, td.menu_id
        FROM transaction_detail td JOIN transaction t ON td.transaction_id = t.id
        WHERE t.status = 'done' AND t.deleted_at IS NULL
    """
    purchase_data = pd.read_sql(purchase_query, db_engine)

    # 4. Muat Interaksi Implisit (Views, Add to Cart)
    implicit_query = """
        SELECT customer_id, menu_id FROM customer_interactions 
        WHERE menu_id IS NOT NULL AND interaction_type IN ('view_detail_menu', 'add_to_cart')
    """
    implicit_data = pd.read_sql(implicit_query, db_engine)
    
    # 5. Gabungkan SEMUA interaksi
    interactions_df = pd.concat([ratings_data, favorites_data, purchase_data, implicit_data], ignore_index=True)
    interactions_df.dropna(subset=['customer_id', 'menu_id'], inplace=True)
    
    # Ambil data menu
    menu_query = """
        SELECT
            menu.id,
            c.name AS canteen_name,
            GROUP_CONCAT(cat.name SEPARATOR ' ') AS categories
        FROM menu
        JOIN canteen c ON menu.canteen_id = c.id
        LEFT JOIN category_menu cm ON menu.id = cm.menu_id
        LEFT JOIN category cat ON cm.category_id = cat.id
        WHERE menu.deleted_at IS NULL
        GROUP BY menu.id, c.name
    """
    menu_df = pd.read_sql(menu_query, db_engine)
    menu_df['main_category'] = menu_df['categories'].fillna('Lainnya').apply(lambda x: x.split()[0])
    
    # Ambil ID dan Email dari tabel customer
    customers_df = pd.read_sql("SELECT id, email FROM customer", db_engine)
    
    # Gabungkan semua data
    merged_df = pd.merge(interactions_df, menu_df, left_on='menu_id', right_on='id')
    merged_df = pd.merge(merged_df, customers_df, left_on='customer_id', right_on='id', suffixes=('_menu', '_customer'))

except Exception as e:
    print(f"Gagal mengambil data: {e}")
    exit()

# --- 3. ANALISIS PERILAKU PENGGUNA ---
print("Menganalisis perilaku setiap pengguna...")
user_analysis = []
for (customer_id, email), group in merged_df.groupby(['customer_id', 'email']):
    total_interactions = len(group)
    
    category_counts = group['main_category'].value_counts(normalize=True)
    top_category = category_counts.index[0]
    top_category_pct = category_counts.iloc[0]
    
    canteen_counts = group['canteen_name'].value_counts(normalize=True)
    top_canteen = canteen_counts.index[0]
    top_canteen_pct = canteen_counts.iloc[0]
    
    persona = "Explorer"
    if top_category in ['Snack', 'Minuman'] and top_category_pct > 0.6:
        persona = "Snacker"
    elif top_canteen_pct > 0.7:
        persona = "Loyalist"

    user_analysis.append({
        "email": email,
        "customer_id": customer_id,
        "persona": persona,
        "total_interactions": total_interactions,
        "top_category": f"{top_category} ({top_category_pct:.0%})",
        "top_canteen": f"{top_canteen} ({top_canteen_pct:.0%})"
    })

# --- 4. TAMPILKAN HASIL ---
analysis_df = pd.DataFrame(user_analysis).sort_values(by="persona")
display_cols = ["email", "persona", "total_interactions", "top_category", "top_canteen"]
print("\n--- Hasil Analisis Persona dari Data Lengkap ---")
print(analysis_df[display_cols].to_string())

print(f"\nContoh ID untuk pengujian kualitatif:")
try:
    snacker = analysis_df[analysis_df['persona'] == 'Snacker'].iloc[0]
    print(f"  -> Persona Snacker: {snacker['email']} (ID: {snacker['customer_id']})")
except IndexError:
    print("  -> Tidak ditemukan persona Snacker.")

try:
    loyalist = analysis_df[analysis_df['persona'] == 'Loyalist'].iloc[0]
    print(f"  -> Persona Loyalist: {loyalist['email']} (ID: {loyalist['customer_id']})")
except IndexError:
    print("  -> Tidak ditemukan persona Loyalist.")