import pandas as pd
import numpy as np
import os
import tempfile
from faker import Faker

fake = Faker()

# -------------------------
# Geração das dimensões
# -------------------------

def generate_customers(num_customers=500):
    data = {
        'customer_id': range(1, num_customers + 1),
        'customer_name': [fake.name() for _ in range(num_customers)],
        'customer_email': [fake.email() for _ in range(num_customers)],
        'customer_since': [fake.date_between(start_date='-5y', end_date='today') for _ in range(num_customers)],
        'country': [fake.country() for _ in range(num_customers)]
    }
    return pd.DataFrame(data)

def generate_products(num_products=100):
    categories = ['Eletrônicos', 'Vestuário', 'Alimentos', 'Móveis', 'Brinquedos']
    data = {
        'product_id': range(1, num_products + 1),
        'product_name': [fake.word().capitalize() for _ in range(num_products)],
        'category': [np.random.choice(categories) for _ in range(num_products)],
        'price': [round(np.random.uniform(10, 1000), 2) for _ in range(num_products)]
    }
    return pd.DataFrame(data)

def generate_employees(num_employees=50):
    departments = ['Vendas', 'Atendimento', 'Financeiro', 'Logística']
    data = {
        'employee_id': range(1, num_employees + 1),
        'employee_name': [fake.name() for _ in range(num_employees)],
        'hire_date': [fake.date_between(start_date='-10y', end_date='today') for _ in range(num_employees)],
        'department': [np.random.choice(departments) for _ in range(num_employees)]
    }
    return pd.DataFrame(data)

def generate_stores(num_stores=20):
    data = {
        'store_id': range(1, num_stores + 1),
        'store_name': [f"Loja {i}" for i in range(1, num_stores + 1)],
        'city': [fake.city() for _ in range(num_stores)],
        'state': [fake.state_abbr() for _ in range(num_stores)]
    }
    return pd.DataFrame(data)

def generate_payment_types():
    payment_methods = ['Cartão de Crédito', 'Cartão de Débito', 'Dinheiro', 'Boleto', 'Pix']
    data = {
        'payment_type_id': range(1, len(payment_methods) + 1),
        'payment_method': payment_methods
    }
    return pd.DataFrame(data)

# -------------------------
# Geração da Fato (Sales)
# -------------------------

def generate_sales(num_sales, customers, products, employees, stores, payment_types):
    data = {
        'sale_id': range(1, num_sales + 1),
        'customer_id': np.random.choice(customers['customer_id'], num_sales),
        'product_id': np.random.choice(products['product_id'], num_sales),
        'employee_id': np.random.choice(employees['employee_id'], num_sales),
        'store_id': np.random.choice(stores['store_id'], num_sales),
        'payment_type_id': np.random.choice(payment_types['payment_type_id'], num_sales),
        'sale_date': [fake.date_between(start_date='-5y', end_date='today') for _ in range(num_sales)],
        'quantity': np.random.randint(1, 10, size=num_sales),
    }

    sales_df = pd.DataFrame(data)

    # Juntar preço do produto para calcular o total_amount
    sales_df = sales_df.merge(products[['product_id', 'price']], on='product_id', how='left')
    sales_df['total_amount'] = sales_df['quantity'] * sales_df['price']
    sales_df.drop(columns=['price'], inplace=True)

    return sales_df

# -------------------------
# Utilidades
# -------------------------

def save_dataframes_to_csv(df_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for name, df in df_dict.items():
        df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)

def measure_avg_row_size(customers, products, employees, stores, payment_types, sample_size=10000):
    """Gera um pequeno sample de vendas e mede o tamanho médio da linha."""
    sales_sample = generate_sales(sample_size, customers, products, employees, stores, payment_types)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
        sales_sample.to_csv(tmp_file.name, index=False)
        tmp_file.flush()
        size_bytes = os.path.getsize(tmp_file.name)

    os.unlink(tmp_file.name)

    avg_row_size = size_bytes / sample_size
    print(f"  Measured average row size: {avg_row_size:.2f} bytes")
    return avg_row_size

def adjust_csv_size(filepath, target_size_mb, tolerance=0.05):
    """Se o CSV ficou maior que o permitido, remove linhas até ajustar."""
    max_size_bytes = target_size_mb * 1024 * 1024 * (1 + tolerance)
    current_size_bytes = os.path.getsize(filepath)

    if current_size_bytes <= max_size_bytes:
        print(f"  CSV already within acceptable size ({current_size_bytes / (1024*1024):.2f}MB)")
        return

    print(f"  CSV too large ({current_size_bytes / (1024*1024):.2f}MB), adjusting...")

    df = pd.read_csv(filepath)

    # Calcula quantas linhas precisam ser removidas
    reduction_ratio = max_size_bytes / current_size_bytes
    new_row_count = int(len(df) * reduction_ratio)

    print(f"  Reducing rows: {len(df)} -> {new_row_count}")

    df = df.iloc[:new_row_count]

    df.to_csv(filepath, index=False)

    final_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  Adjusted CSV size: {final_size_mb:.2f}MB")

def generate_and_save_synthetic_data(target_size_mb_list, base_output_dir='', tolerance=0.05):
    for target_size in target_size_mb_list:
        print(f"\nGenerating dataset for ~{target_size}MB...")

        # Criar mini-dimensões para medir
        customers_sample = generate_customers(100)
        products_sample = generate_products(50)
        employees_sample = generate_employees(20)
        stores_sample = generate_stores(10)
        payment_types_sample = generate_payment_types()

        # Medir tamanho médio
        avg_row_size = measure_avg_row_size(customers_sample, products_sample, employees_sample, stores_sample, payment_types_sample)

        target_bytes = target_size * 1024 * 1024
        estimated_rows = int(target_bytes / avg_row_size)

        print(f"  Estimated number of rows: {estimated_rows}")

        # Agora gera as dimensões reais
        num_customers = max(100, int(estimated_rows * 0.01))
        num_products = max(50, int(estimated_rows * 0.005))
        num_employees = max(20, int(estimated_rows * 0.002))
        num_stores = max(10, int(estimated_rows * 0.001))

        customers = generate_customers(num_customers)
        products = generate_products(num_products)
        employees = generate_employees(num_employees)
        stores = generate_stores(num_stores)
        payment_types = generate_payment_types()

        sales = generate_sales(estimated_rows, customers, products, employees, stores, payment_types)

        # Salvar
        output_dir = os.path.join(base_output_dir, f"{target_size}MB")
        save_dataframes_to_csv({
            'customers': customers,
            'products': products,
            'employees': employees,
            'stores': stores,
            'payment_types': payment_types,
            'sales': sales
        }, output_dir)

        sales_filepath = os.path.join(output_dir, 'sales.csv')

        adjust_csv_size(sales_filepath, target_size, tolerance)

        print(f"  Saved dataset for ~{target_size}MB at {output_dir}")

# -------------------------
# Execução
# -------------------------

if __name__ == "__main__":
    target_sizes_mb = [10, 100, 1000, 10000, 50000]
    generate_and_save_synthetic_data(target_sizes_mb)