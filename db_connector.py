import mysql.connector
from mysql.connector import Error


class Connector:
    def __init__(self, host_name='localhost', user_name='root', user_password='password', database='connect4'):
        self.connection = None
        try:
            self.connection = mysql.connector.connect(
                host=host_name,
                user=user_name,
                passwd=user_password,
                database=database,
            )
        except Error as e:
            print(f"The error '{e}' occurred")

    def send_query(self, query, print_out=True):
        cursor = self.connection.cursor()
        try:
            cursor.execute(query)
            result = cursor.fetchall()
            self.connection.commit()
            if print_out:
                print('send query successfully')
            return result
        except Error as e:
            print(f"The error '{e}' occurred")


if __name__ == '__main__':
    db = Connector()
    print(db.send_query('SELECT * FROM parameters LIMIT 10'))
