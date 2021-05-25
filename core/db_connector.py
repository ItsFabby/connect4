import mysql.connector
from mysql.connector import Error
import numpy as np
import pandas as pd
from typing import Optional

import constants as c


class Connector:
    """
        Allows inserting and retrieving training data from a MySQL database.
    """

    def __init__(self):
        self.connection = None
        try:
            self.connection = mysql.connector.connect(
                host=c.HOST_NAME,
                user=c.USER_NAME,
                passwd=c.PASSWORD,
                database=c.DATABASE,
            )
        except Error as e:
            print(f"The error '{e}' occurred")

    def _send_query(self, query: str, print_out: bool = False, print_out_errors: bool = True) -> Optional[list]:
        cursor = self.connection.cursor()
        try:
            cursor.execute(query)
            result = cursor.fetchall()
            self.connection.commit()
            if print_out:
                print('send query successfully')
            return result
        except Error as e:
            if print_out_errors:
                print(f"The error '{e}' occurred")

    def retrieve_data(self, query: str) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_sql(query, self.connection)
            self.connection.commit()
            return df
        except Error as e:
            print(f"The error '{e}' occurred")

    # column number still hardcoded
    def insert_examples(self, examples: list, count_factor: float = 1.0, table: str = 'training_data') -> None:
        for ex in examples:
            ex = self._example_to_dict(ex)
            data = self.retrieve_data(f"SELECT * FROM {table} WHERE state = '{ex['state']}'")
            if not data.empty:
                old_dict = data.to_dict('records')[0]
                ex = self._merge_examples(old_dict, ex, count_factor=count_factor)
                self._send_query(
                    f"UPDATE {table} SET state = '{ex['state']}', val = {ex['val']},"
                    f"p0 = {ex['p0']}, p1 = {ex['p1']}, p2 = {ex['p2']}, p3 = {ex['p3']},"
                    f"p4 = {ex['p4']}, p5 = {ex['p5']}, p6 = {ex['p6']},"
                    f"counter = {ex['counter']}, last_edited = CURRENT_TIMESTAMP WHERE state = '{ex['state']}';"
                )
            else:
                self._send_query(
                    f"INSERT INTO {table} (state, val, p0, p1, p2, p3, p4, p5, p6, counter)"
                    f" VALUES ('{ex['state']}', {ex['val']}, {ex['p0']}, {ex['p1']}, {ex['p2']}, {ex['p3']},"
                    f"{ex['p4']}, {ex['p5']}, {ex['p6']}, 1);"
                )

    @staticmethod
    def _merge_examples(old_dict: dict, new_dict: dict, count_factor: float = 1.0) -> dict:
        counter = old_dict['counter']
        scaled_counter = counter * count_factor
        merged_dict = {'state': old_dict['state'], 'counter': counter + 1}
        for column in ['val'] + [f'p{i}' for i in range(c.COLUMNS)]:
            merged_dict.update({
                column: (1 / (scaled_counter + 1)) * new_dict[column] +
                        (scaled_counter / (scaled_counter + 1)) * old_dict[column]
            })
        return merged_dict

    def df_to_examples(self, df: pd.DataFrame) -> list:
        dictionary = df.to_dict('records')
        examples = []
        for row in dictionary:
            state = self._string_to_state(row['state'])
            value = row['val']
            pi = np.array([row[f'p{i}'] for i in range(c.COLUMNS)])
            examples.append((state, (pi, value)))
        return examples

    def _example_to_dict(self, example: tuple) -> dict:
        state = self._state_to_string(example[0])
        value = example[1][1]
        pi = example[1][0]
        return {'state': state, 'val': value, **dict((f'p{i}', pi[i]) for i in range(7))}

    @staticmethod
    def _state_to_string(state: np.array) -> str:
        return ''.join(str(int(state[i, j] + 1)) for i, j in np.ndindex(np.shape(state)))

    @staticmethod
    def _string_to_state(string: str) -> np.array:
        return np.reshape(np.array([int(char) for char in string]), (c.COLUMNS, c.ROWS)) - 1

    def _create_training_data_table(self, table: str) -> None:
        self._send_query(
            f"CREATE TABLE IF NOT EXISTS {table} ("
            f"id INT(10) NOT NULL AUTO_INCREMENT,"
            f"state VARCHAR(42) UNIQUE NOT NULL,"
            f"val DECIMAL(7,6) NOT NULL,"
            f"p0 DECIMAL (7,6) NOT NULL,"
            f"p1 DECIMAL (7,6) NOT NULL,"
            f"p2 DECIMAL (7,6) NOT NULL,"
            f"p3 DECIMAL (7,6) NOT NULL,"
            f"p4 DECIMAL (7,6) NOT NULL,"
            f"p5 DECIMAL (7,6) NOT NULL,"
            f"p6 DECIMAL (7,6) NOT NULL,"
            f"counter INT(10) NOT NULL"
            f"last_edited TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
            f"PRIMARY KEY (id)"
            f");"
        )
        self._send_query(
            f"CREATE UNIQUE INDEX state_index ON {table}(state);",
            print_out_errors=False
        )
