from tkinter import *


main_menu = Tk()
w = Button (main_menu, text='Record').grid(row=0)
Label(main_menu, text='Last Name').grid(row=1)


e1 = Entry(main_menu)
e2 = Entry(main_menu)


e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
mainloop()