import tkinter as tk
from tkinter import filedialog, ttk
import csv


class CSVExcelApp(tk.Tk):
    """Simple Excel-like interface for viewing CSV data."""

    def __init__(self):
        super().__init__()
        self.title("Mini Excel")
        self.geometry("800x600")
        self._create_widgets()

    def _create_widgets(self):
        toolbar = tk.Frame(self, bd=1, relief=tk.RAISED)
        import_btn = tk.Button(toolbar, text="Import CSV", command=self.load_csv)
        import_btn.pack(side=tk.LEFT, padx=2, pady=2)
        clear_btn = tk.Button(toolbar, text="Clear", command=self.clear_table)
        clear_btn.pack(side=tk.LEFT, padx=2, pady=2)
        exit_btn = tk.Button(toolbar, text="Exit", command=self.quit)
        exit_btn.pack(side=tk.LEFT, padx=2, pady=2)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.table = ttk.Treeview(self)
        self.table.pack(expand=True, fill=tk.BOTH)

        vsb = ttk.Scrollbar(self.table, orient="vertical", command=self.table.yview)
        hsb = ttk.Scrollbar(self.table, orient="horizontal", command=self.table.xview)
        self.table.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")

    def clear_table(self):
        self.table.delete(*self.table.get_children())
        self.table["columns"] = ()
        self.table["show"] = "tree"

    def load_csv(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not file_path:
            return
        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            self._populate_table(reader)

    def _populate_table(self, reader):
        self.clear_table()
        rows = list(reader)
        if not rows:
            return
        headers = rows[0]
        self.table["columns"] = headers
        self.table["show"] = "headings"
        for col in headers:
            self.table.heading(col, text=col)
            self.table.column(col, width=100, anchor=tk.W)
        for row in rows[1:]:
            self.table.insert("", tk.END, values=row)


if __name__ == "__main__":
    app = CSVExcelApp()
    app.mainloop()
