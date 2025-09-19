import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from urllib.parse import quote_plus
import os
import uuid
import random
import numpy as np

print("--- Memulai Skrip Simulasi Data Interaksi ---")

# --- 1. SETUP KONEKSI DATABASE ---
load_dotenv()
DB_USER = os.environ.get('DB_USERNAME')
DB_PASSWORD = os.environ.get('DB_PASSWORD')
DB_HOST = os.environ.get('DB_HOST')
DB_PORT = os.environ.get('DB_PORT')
DB_NAME = os.environ.get('DB_DATABASE')

encoded_password = quote_plus(DB_PASSWORD)
db_connection_str = f'mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
db_engine = create_engine(db_connection_str)

# --- 2. AMBIL DATA MASTER DARI DATABASE ---
print("Mengambil data master (menu, customer)...")
try:
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
    menu_df['categories'] = menu_df['categories'].fillna('Lainnya')

    customers_df = pd.read_sql("SELECT id FROM customer", db_engine)
    customer_ids = customers_df['id'].tolist()
    
    if menu_df.empty or not customer_ids:
        print("FATAL: Data menu atau customer tidak ditemukan. Pastikan database terisi data master.")
        exit()

except Exception as e:
    print(f"Gagal mengambil data master: {e}")
    exit()

# --- 3. DEFINISI PERSONA & KONFIGURASI SIMULASI ---
TOTAL_USERS_TO_SIMULATE = 20
personas = {
    "snacker": {
        "interaction_range": (15, 25),
        "category_weights": {"Minuman": 0.4, "Snack": 0.3, "Makanan": 0.2, "Lainnya": 0.1}
    },
    "loyalist": {
        "interaction_range": (8, 15),
        "canteen_loyal_to": menu_df['canteen_name'].unique()[:2].tolist(), # Ambil 2 kantin pertama
        "canteen_loyalty_prob": 0.8
    },
    "explorer": {
        "interaction_range": (5, 12),
    }
}

# --- 4. PROSES GENERASI DATA ---
print(f"Memulai generasi data untuk {TOTAL_USERS_TO_SIMULATE} pengguna...")
all_new_ratings = []
all_new_favorites = []

users_to_simulate = random.sample(customer_ids, min(TOTAL_USERS_TO_SIMULATE, len(customer_ids)))

for customer_id in users_to_simulate:
    persona_name = random.choice(list(personas.keys()))
    persona = personas[persona_name]
    num_interactions = random.randint(*persona['interaction_range'])
    
    menus_to_interact = []
    if persona_name == "snacker":
        cat_list = list(persona['category_weights'].keys())
        prob_list = list(persona['category_weights'].values())
        chosen_categories = np.random.choice(cat_list, size=num_interactions, p=prob_list)
        for cat in chosen_categories:
            possible_menus = menu_df[menu_df['categories'].str.contains(cat, case=False, na=False)]
            if not possible_menus.empty:
                menus_to_interact.append(possible_menus.sample(1).iloc[0])
    elif persona_name == "loyalist":
        for _ in range(num_interactions):
            if random.random() < persona['canteen_loyalty_prob']:
                possible_menus = menu_df[menu_df['canteen_name'].isin(persona['canteen_loyal_to'])]
            else:
                possible_menus = menu_df[~menu_df['canteen_name'].isin(persona['canteen_loyal_to'])]
            if not possible_menus.empty:
                menus_to_interact.append(possible_menus.sample(1).iloc[0])
    else: # Explorer
        menus_to_interact = [menu_df.sample(1).iloc[0] for _ in range(num_interactions)]

    for menu in menus_to_interact:
        if random.random() < 0.7:
            all_new_ratings.append({
                'id': str(uuid.uuid4()),
                'customer_id': customer_id,
                'menu_id': menu['id'],
                'rating': random.randint(3, 5),
                'comment': '',
                'transaction_detail_id': str(uuid.uuid4())
            })
        else:
            all_new_favorites.append({
                'id': str(uuid.uuid4()),
                'customer_id': customer_id,
                'menu_id': menu['id'],
            })
            
print("Generasi data selesai.")

# --- 5. MASUKKAN DATA KE DATABASE ---
if all_new_ratings:
    ratings_df = pd.DataFrame(all_new_ratings).drop_duplicates(subset=['customer_id', 'menu_id'])
    print(f"Menyimpan {len(ratings_df)} data rating baru ke database...")
    try:
        with db_engine.connect() as connection:
            connection.execute(text("SET FOREIGN_KEY_CHECKS=0;"))
            ratings_df.to_sql('menu_rating', connection, if_exists='append', index=False)
            connection.execute(text("SET FOREIGN_KEY_CHECKS=1;"))
            connection.commit() # Pastikan perubahan tersimpan
    except Exception as e:
        print(f"Gagal menyimpan ratings: {e}")

if all_new_favorites:
    favorites_df = pd.DataFrame(all_new_favorites).drop_duplicates(subset=['customer_id', 'menu_id'])
    print(f"Menyimpan {len(favorites_df)} data favorit baru ke database...")
    try:
        favorites_df.to_sql('favorite_menu', db_engine, if_exists='append', index=False)
    except Exception as e:
        print(f"Gagal menyimpan favorites: {e}")
    
print("\n--- Simulasi Selesai ---")
print("Database Anda sekarang memiliki data interaksi baru. Anda siap untuk menjalankan 'evaluate.py'.")