import mysql.connector
from mysql.connector import Error
import numpy as np
import pandas as pd


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

    def send_query(self, query, print_out=False):
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

    def retrieve_data(self, query):
        try:
            df = pd.read_sql(query, self.connection)
            self.connection.commit()
            return df
        except Error as e:
            print(f"The error '{e}' occurred")

    def insert_examples(self, examples, count_factor=1.0, table='training_data'):
        for ex in examples:
            ex = self.example_to_dict(ex)
            data = self.retrieve_data(f"SELECT * FROM {table} WHERE state = '{ex['state']}'")
            if not data.empty:
                old_dict = data.to_dict('records')[0]
                ex = self.merge_examples(old_dict, ex, count_factor=count_factor)
                self.send_query(
                    f"UPDATE {table} SET state = '{ex['state']}', val = {ex['val']},"
                    f"p0 = {ex['p0']}, p1 = {ex['p1']}, p2 = {ex['p2']}, p3 = {ex['p3']},"
                    f"p4 = {ex['p4']}, p5 = {ex['p5']}, p6 = {ex['p6']},"
                    f"counter = {ex['counter']}, last_edited = CURRENT_TIMESTAMP WHERE state = '{ex['state']}';"
                )
            else:
                self.send_query(
                    f"INSERT INTO {table} (state, val, p0, p1, p2, p3, p4, p5, p6, counter)"
                    f" VALUES ('{ex['state']}', {ex['val']}, {ex['p0']}, {ex['p1']}, {ex['p2']}, {ex['p3']},"
                    f"{ex['p4']}, {ex['p5']}, {ex['p6']}, 1);"
                )

    @staticmethod
    def merge_examples(old_dict, new_dict, count_factor=1.0):
        counter = old_dict['counter']
        scaled_counter = counter * count_factor
        merged_dict = {'state': old_dict['state'], 'counter': counter + 1}
        for column in ['val'] + [f'p{i}' for i in range(7)]:
            merged_dict.update({column: (1 / (scaled_counter + 1))
                                * new_dict[column] + (scaled_counter / (scaled_counter + 1)) * old_dict[column]})
        return merged_dict

    def df_to_examples(self, df):
        dictionary = df.to_dict('records')
        examples = []
        for row in dictionary:
            state = self.string_to_state(row['state'])
            value = row['val']
            pi = np.array([row[f'p{i}'] for i in range(7)])
            examples.append((state, (pi, value)))
        return examples

    def example_to_dict(self, example):
        state = self.state_to_string(example[0])
        value = example[1][1]
        pi = example[1][0]
        return {'state': state, 'val': value, **dict((f'p{i}', pi[i]) for i in range(7))}

    @staticmethod
    def state_to_string(state):
        return ''.join(str(int(state[i, j] + 1)) for i, j in np.ndindex(np.shape(state)))

    @staticmethod
    def string_to_state(string):
        return np.reshape(np.array([int(c) for c in string]), (7, 6)) - 1


if __name__ == '__main__':
    db = Connector()
    print(db.df_to_examples(db.retrieve_data('select * from training_data limit 3')))
