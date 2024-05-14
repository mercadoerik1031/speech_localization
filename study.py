import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('snn_study.db')

# Create a cursor object
cursor = conn.cursor()

# Query to find all tables in the database
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

# Fetch all table names
tables = cursor.fetchall()

# Loop through the tables and print their contents
for table_name in tables:
    table_name = table_name[0]  # correct format
    print(f"Table: {table_name}")
    cursor.execute(f"SELECT * FROM {table_name}")
    records = cursor.fetchall()
    for record in records:
        print(record)
    print("\n")  # Adds a newline between tables for better readability

# Close the connection
conn.close()