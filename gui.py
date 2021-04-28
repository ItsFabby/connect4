import tkinter as tk


class Board(tk.Frame):
    def __init__(self, parent, game, rows, columns, size=32):
        self.rows = rows
        self.columns = columns
        self.size = size
        self.game = game

        canvas_width = columns * size
        canvas_height = rows * size

        tk.Frame.__init__(self, parent)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0,
                                width=canvas_width, height=canvas_height, background="darkgrey")
        self.canvas.pack(side="top", fill="both", expand=True, padx=2, pady=2)

        self.canvas.bind("<Configure>", self.refresh)

    def refresh(self, event):
        # Redraw the board, possibly in response to window being resized
        xsize = int((event.width - 1) / self.columns)
        ysize = int((event.height - 1) / self.rows)
        self.size = min(xsize, ysize)
        self.redraw()

    def redraw(self):
        self.canvas.delete("square")
        for row in range(self.rows):
            for col in range(self.columns):
                x1 = (col * self.size)
                y1 = ((self.rows-row-1) * self.size)
                x2 = x1 + self.size
                y2 = y1 + self.size
                if self.game.game_state[col][row] == 0:
                    self.canvas.create_rectangle(x1, y1, x2, y2, outline="black", fill="white", tags="square")
                elif self.game.game_state[col][row] == 1:
                    self.canvas.create_rectangle(x1, y1, x2, y2, outline="black", fill="red", tags="square")
                else:
                    self.canvas.create_rectangle(x1, y1, x2, y2, outline="black", fill="blue", tags="square")
