import Tkinter as tk





class MainWindow(tk.Frame):
	def __init__(self, *args, **kwargs):
		tk.Frame.__init__(self, *args, **kwargs)
		self.trainingBtn = tk.Button(self, text="Training", command=self.create_training_window)
		self.trainingBtn.pack(side="top")
		
	def create_training_window(self):
		t = tk.Toplevel(self)
		t.wm_title("Training window")
		t.pack(fill=BOTH, expand=True)
        
        frame1 = Frame(self)
        frame1.pack(fill=X)
        
        lbl1 = Label(frame1, text="Title", width=6)
        lbl1.pack(side=LEFT, padx=5, pady=5)           
       
        entry1 = Entry(frame1)
        entry1.pack(fill=X, padx=5, expand=True)
        
        frame2 = Frame(self)
        frame2.pack(fill=X)
        
        lbl2 = Label(frame2, text="Author", width=6)
        lbl2.pack(side=LEFT, padx=5, pady=5)        

        entry2 = Entry(frame2)
        entry2.pack(fill=X, padx=5, expand=True)
        
        frame3 = Frame(self)
        frame3.pack(fill=BOTH, expand=True)
        
        lbl3 = Label(frame3, text="Review", width=6)
        lbl3.pack(side=LEFT, anchor=N, padx=5, pady=5)        

        txt = Text(frame3)
        txt.pack(fill=BOTH, pady=5, padx=5, expand=True)  


		# oversampling_frame = tk.Frame(t)
		# oversampling_frame.pack(fill=tk.BOTH, expand=True)

		# l = tk.Label(oversampling_frame, text="test")
		# t_btn = tk.Button(oversampling_frame, text="True", width=12)
		# l.pack(side="left", fill="both", expand=True, padx=100, pady=100)
		# t_btn.pack(side='right', pady=5)


		# generate_frame = tk.Frame(t)
		# generate_frame.pack(fill=tk.BOTH, expand=True)
		# l = tk.Label(oversampling_frame, text="test")
		# t_btn = tk.Button(oversampling_frame, text="True", width=12)
		# l.pack(side="left", fill="both", expand=True, padx=100, pady=100)
		# t_btn.pack(side='right', pady=5)
		

		
	



if __name__ == "__main__":
    root = tk.Tk()
    main = MainWindow(root)
    main.pack(side="top", fill="both", expand=True, padx=100, pady=100)
    root.mainloop()