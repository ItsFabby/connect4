import tkinter as tk
from threading import Thread
import math
from typing import Any

from game import Game
import constants as c


class GUI(tk.Frame):
    def __init__(self):
        super().__init__()
        self.game = Game()

        """GUI elements"""
        self.header = tk.Label(self)
        self.header.grid(row=0, column=0)

        self.field = tk.Canvas(self, borderwidth=0, highlightthickness=0,
                               width=c.COLUMNS * c.SQUARE_SIZE, height=c.ROWS * c.SQUARE_SIZE)
        self.field.grid(row=1, column=0, padx=2, pady=2)

        self._restart_button = tk.Button(self, text='Restart game', command=self._restart)
        self._restart_button.grid(row=3, column=0)

        self.difficulty_frame = tk.Frame(self)
        self.difficulty_frame.grid(row=4, column=0)
        self.difficulty = tk.IntVar(self)
        self.difficulty.set(50)
        self.difficulty_label = tk.Label(self.difficulty_frame,
                                         text='Number of evaluated positions by the neural network per turn:')
        self.difficulty_label.grid(row=0, column=1)
        self.difficulty_box = tk.OptionMenu(self.difficulty_frame, self.difficulty, *[10, 25, 50, 75, 100, 150, 200])
        self.difficulty_box.grid(row=0, column=2)

        self.player_selection = tk.Frame(self)
        self.player_selection.grid(row=5, column=0)

        self.player1 = tk.StringVar(self)
        self.player1.set('Human')
        self.player1_label = tk.Label(self.player_selection, text='Player 1 (red): ')
        self.player1_label.grid(row=0, column=0)
        self.player1_box = tk.OptionMenu(self.player_selection, self.player1, *['Human', 'Neural Network'],
                                         command=self._option_trigger)
        self.player1_box.grid(row=0, column=1)

        self.player2 = tk.StringVar(self)
        self.player2.set('Neural Network')
        self.player2_label = tk.Label(self.player_selection, text='Player 2 (blue): ')
        self.player2_label.grid(row=0, column=3)
        self.player2_box = tk.OptionMenu(self.player_selection, self.player2, *['Human', 'Neural Network'],
                                         command=self._option_trigger)
        self.player2_box.grid(row=0, column=4)

        self.pack()

        """Starting event loop"""
        self.field.bind('<Button-1>', self._move_event)
        self._check_ai()
        self._redraw()
        self.mainloop()

    def _option_trigger(self, _event: Any) -> None:
        # overwriting the current game to prevent still ongoing threads from interacting
        state = self.game.game_state
        player = self.game.player
        finished = self.game.finished
        self.game = Game()
        self.game.game_state = state
        self.game.player = player
        self.game.finished = finished

        self._check_ai()

    def _restart(self) -> None:
        self.game = Game()
        self._redraw()
        self._update_header()
        self._check_ai()

    def _move_event(self, event: Any) -> None:
        if not self._is_nnet(self.game.player) and not self.game.finished:
            column = math.floor(event.x / c.SQUARE_SIZE)
            self.game.make_move(column)
            self._redraw()
            self.update()
            self._check_ai()

    def _is_nnet(self, player: int) -> bool:
        if player == 1:
            return self.player1.get() == 'Neural Network'
        else:
            return self.player2.get() == 'Neural Network'

    def _check_ai(self) -> None:
        self._update_header()
        if self._is_nnet(self.game.player) and not self.game.finished:
            Thread(target=self._ai_move).start()

    def _ai_move(self) -> None:
        self.game.make_move(self.game.decide_move('nnet', iterations=self.difficulty.get()))
        self._redraw()
        self._check_ai()

    def _update_header(self) -> None:
        if not self.game.finished:
            self.header.config(text=f"{'Red' if self.game.player == 1 else 'Blue'}'s turn: "
                                    f"{'The AI is calculating.' if self._is_nnet(self.game.player) else 'Make a move!'}"
                               )
        else:
            self.header.config(text=f"{'Red' if self.game.winner == 1 else 'Blue'} has won!")

    def _redraw(self) -> None:
        self.field.delete("square")
        for row in range(c.ROWS):
            for col in range(c.COLUMNS):
                x1 = (col * c.SQUARE_SIZE)
                y1 = ((c.ROWS - row - 1) * c.SQUARE_SIZE)
                x2 = x1 + c.SQUARE_SIZE
                y2 = y1 + c.SQUARE_SIZE
                if self.game.game_state[col][row] == 0:
                    self.field.create_rectangle(x1, y1, x2, y2, outline="black", fill="white", tags="square")
                elif self.game.game_state[col][row] == 1:
                    self.field.create_rectangle(x1, y1, x2, y2, outline="black", fill="red", tags="square")
                else:
                    self.field.create_rectangle(x1, y1, x2, y2, outline="black", fill="blue", tags="square")


if __name__ == '__main__':
    GUI()
