import pymysql


def show_databases():
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='12345'
    )

    cursor = conn.cursor()
    cursor.execute("SHOW DATABASES")

    for db in cursor.fetchall():
        print(db)