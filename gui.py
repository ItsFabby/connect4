import tkinter as tk
from threading import Thread
import math
from game import Game


class Board(tk.Frame):
    def __init__(self, rows=6, columns=7, size=64):
        super().__init__()
        self.rows = rows
        self.columns = columns
        self.size = size
        self.game = Game()

        canvas_width = columns * size
        canvas_height = rows * size

        self.header = tk.Label(self)
        self.header.grid(row=0, column=0)

        self.field = tk.Canvas(self, borderwidth=0, highlightthickness=0,
                               width=canvas_width, height=canvas_height)
        self.field.grid(row=1, column=0, padx=2, pady=2)

        self.restart_button = tk.Button(self, text='Restart game', command=self.restart)
        self.restart_button.grid(row=3, column=0)

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
        self.player1.set('human')
        self.player1_label = tk.Label(self.player_selection, text='Player 1 (red): ')
        self.player1_label.grid(row=0, column=0)
        self.player1_box = tk.OptionMenu(self.player_selection, self.player1, *['human', 'neural network'],
                                         command=self.option_trigger)
        self.player1_box.grid(row=0, column=1)

        self.player2 = tk.StringVar(self)
        self.player2.set('neural network')
        self.player2_label = tk.Label(self.player_selection, text='Player 2 (blue): ')
        self.player2_label.grid(row=0, column=3)
        self.player2_box = tk.OptionMenu(self.player_selection, self.player2, *['human', 'neural network'],
                                         command=self.option_trigger)
        self.player2_box.grid(row=0, column=4)

        self.pack()

        # self.bind("<Configure>", self.refresh)
        self.field.bind('<Button-1>', self.move_event)

        self.check_ai()
        self.redraw()
        self.mainloop()

    def option_trigger(self, event):
        # overwriting the current game to prevent still ongoing threads from interacting
        state = self.game.game_state
        player = self.game.player
        finished = self.game.finished
        self.game = Game()
        self.game.game_state = state
        self.game.player = player
        self.game.finished = finished

        self.check_ai()

    def refresh(self, event):
        # Redraw the board, possibly in response to window being resized
        xsize = int((event.width - 1) / self.columns)
        ysize = int((event.height - 1) / self.rows)
        self.size = min(xsize, ysize)
        self.redraw()

    def restart(self):
        self.game = Game()
        self.redraw()
        self.update_header()
        self.check_ai()

    def move_event(self, event):
        if not self.is_nnet(self.game.player) and not self.game.finished:
            column = math.floor(event.x / self.size)
            self.game.make_move(column)
            self.redraw()
            self.update()
            self.check_ai()

    def is_nnet(self, player):
        if player == 1:
            return self.player1.get() == 'neural network'
        else:
            return self.player2.get() == 'neural network'

    def check_ai(self):
        self.update_header()
        if self.is_nnet(self.game.player) and not self.game.finished:
            Thread(target=self.ai_move).start()

    def ai_move(self):
        print(self.difficulty.get())
        self.game.make_move(self.game.decide_move('nnet', runs=self.difficulty.get()))
        self.redraw()
        self.check_ai()

    def update_header(self):
        if not self.game.finished:
            self.header.config(text=f"{'Red' if self.game.player == 1 else 'Blue'}'s turn: "
                                    f"{'The AI is calculating.' if self.is_nnet(self.game.player) else 'Make a move!'}"
                               )
        else:
            self.header.config(text=f"{'Red' if self.game.winner == 1 else 'Blue'} has won!")

    def redraw(self):
        self.field.delete("square")
        for row in range(self.rows):
            for col in range(self.columns):
                x1 = (col * self.size)
                y1 = ((self.rows - row - 1) * self.size)
                x2 = x1 + self.size
                y2 = y1 + self.size
                if self.game.game_state[col][row] == 0:
                    self.field.create_rectangle(x1, y1, x2, y2, outline="black", fill="white", tags="square")
                elif self.game.game_state[col][row] == 1:
                    self.field.create_rectangle(x1, y1, x2, y2, outline="black", fill="red", tags="square")
                else:
                    self.field.create_rectangle(x1, y1, x2, y2, outline="black", fill="blue", tags="square")
