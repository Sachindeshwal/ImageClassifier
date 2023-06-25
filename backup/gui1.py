import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy
from keras.models import load_model

model = load_model('cifar10_model.h5')

classes = {
    0: 'Aeroplane',
    1: 'Automobile',
    2: 'Bird',
    3: 'Cat' ,
    4: 'Deer',
    5: 'Dog',
    6: 'Frog',
    7: 'Horse',
    8: 'Ship',
    9: 'Truck',
}

def uploaded_image():
    file_path = filedialog.askopenfilename()
    uploaded = Image.open(file_path)
    uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
    im = ImageTk.PhotoImage(uploaded)
    sign_image.configure(image = im)
    sign_image.image = im
    label.configure(text = ' ')
    show_classify_button(file_path)

def show_classify_button(file_path):
    classify_btn = Button(top, text = "classify Image", command = lambda: classify(file_path), padx = 10, pady = 5)
    classify_btn.configure(background = "#364156", foreground = "white", font = ('arial', 10, 'bold'))
    classify_btn.place(relx = 0.79, rely = 0.46)

def classify(file_path):
    image = Image.open(file_path)
    image = image.resize((32, 32))
    image = numpy.expand_dims(image, axis = 0)
    image = numpy.array(image)
    pred = model.predict([image])[0]
    c = numpy.argmax(pred) 
    #sign = classes[pred]
    print(c)
    label.configure(foreground = "#011638", text = classes[c])
    

top = tk.Tk()
top.geometry('800x600')
top.title("Image classification")
top.configure(background = "#CDCDCD")

heading = Label(top, text = "Image classification", pady = 20, font = ('arial', 20, 'bold'))
heading.configure(background = "#CDCDCD", foreground = '#364156')
heading.pack()

upload = Button(top, text = "Upload image", command = uploaded_image, padx = 10, pady = 5)
upload.configure(background = "#364156", foreground = "white",  font = ('arial', 10, 'bold'))
upload.pack(side = BOTTOM, pady = 50)

#uploaded image
sign_image = Label(top)
sign_image.pack(side = BOTTOM, expand = True)

#predicted class
label = Label(top, background = "#CDCDCD", font = ('arial', 15, 'bold'))
label.pack(side = BOTTOM, expand = True)



top.mainloop()
