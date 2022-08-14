import mysql.connector

db = mysql.connector.connect(user='root', password='Hj1374@@',
                             host='localhost', database='pythonmysql',
                             auth_plugin='mysql_native_password')
print(db)

cursor = db.cursor()
SQL = "SHOW DATABASES"

cursor.execute(SQL)
databases = cursor.fetchall()

for database in databases:
    print(database)

db.close()


