import numpy as np
import pandas as pd
import pymongo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import warnings
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import re
import threading
import time
import pickle as pk

warnings.filterwarnings("ignore", category=UserWarning)
lst = [['Age', 'Gender', 'BMI', 'Children', 'Smoker', 'Region', 'Charges', 'InsuranceID']]
def callback(event):
    global selected_items
    if event.state & 4:  # Check if Ctrl key is pressed
        if event.keysym == "a":
            # Select all records when Ctrl+A is pressed
            selected_items = tree.get_children()
            for item in selected_items:
                tree.selection_add(item)
    else:
        selected_items = tree.selection()
        for item in selected_items:
            tree.tag_configure("selected", background="white")
            tree.item(item, tags=("selected",))

    # Rest of the callback function code
    if selected_items:
        item = selected_items[0]
        values = tree.item(item, 'values')
        ID_label2.set(values[7])  # InsuranceID
        age_var.set(values[0])  # Age
        sex_var.set(values[1])  # Gender
        bmi_var.set(values[2])  # BMI
        children_var.set(values[3])  # Children
        smoker_var.set(values[4])  # Smoker
        region_var.set(values[5])  # Region
        charges_prediction.set(values[6])  # Charges

# Bind the callback function to the Treeview selection event
    tree.bind('<ButtonRelease-1>', callback)
    tree.bind('<Control-a>', callback)
def delete():
    global mycol, tree, selected_items
    selected_items = tree.selection()  # Get all selected items

    if not selected_items:
        return

    r = messagebox.askokcancel("Delete?", "Do you want to delete the selected records (will be backed up for 30 mins)?")
    if r:
        current_time = time.time()
        for item in selected_items:
            values = tree.item(item, 'values')
            insurance_id = values[7]  # InsuranceID

            # Store the record in InsuranceInfo2 table
            deleted_record = mycol.find_one({"InsuranceID": int(insurance_id)})
            deleted_record["_deleted_at"] = current_time
            mydb["InsuranceInfo2"].insert_one(deleted_record)

            # Schedule the permanent deletion after 30 minutes
            def schedule_permanent_deletion(insurance_id):
                time.sleep(1800)  # 30 minutes
                mydb["InsuranceInfo2"].delete_one({"InsuranceID": insurance_id})

            deletion_thread = threading.Thread(target=schedule_permanent_deletion, args=(int(insurance_id),))
            deletion_thread.start()

            myquery = {"InsuranceID": int(insurance_id)}
            mycol.delete_one(myquery)
            tree.delete(item)  # Delete the selected record from the table

        # Clear the input fields
        age_var.set("")
        sex_var.set("Female")
        bmi_var.set("")
        children_var.set("")
        smoker_var.set("Yes")
        region_var.set("Southeast")
        charges_prediction.set("")

        # Update the Treeview to reflect the changes
        refresh_table()

        selected_items = []

# Bind the delete function to the Delete key event
    tree.bind(delete)
def restore_deleted_records():
    restore_window = tk.Toplevel()
    restore_window.title('Restore Deleted Records')

    # Create a Treeview to display the deleted records
    restore_tree = ttk.Treeview(restore_window, columns=('Age', 'Gender', 'BMI', 'Children', 'Smoker', 'Region', 'Charges', 'InsuranceID'))
    restore_tree.heading('#0', text='Index')
    restore_tree.heading('Age', text='Age')
    restore_tree.heading('Gender', text='Gender')
    restore_tree.heading('BMI', text='BMI')
    restore_tree.heading('Children', text='Children')
    restore_tree.heading('Smoker', text='Smoker')
    restore_tree.heading('Region', text='Region')
    restore_tree.heading('Charges', text='Charges')
    restore_tree.heading('InsuranceID', text='InsuranceID')

    restore_tree.column('#0', width=50)
    restore_tree.column('Age', width=50)
    restore_tree.column('Gender', width=100)
    restore_tree.column('BMI', width=100)
    restore_tree.column('Children', width=80)
    restore_tree.column('Smoker', width=80)
    restore_tree.column('Region', width=100)
    restore_tree.column('Charges', width=100)
    restore_tree.column('InsuranceID', width=100)
    restore_tree.grid(row=0, column=0, padx=10, pady=10, rowspan=2)
    # Populate the Treeview with deleted records
    deleted_records = mydb["InsuranceInfo2"].find({})
    for record in deleted_records:
        if "_deleted_at" in record:
            deleted_time = record["_deleted_at"]
            current_time = time.time()
            if current_time - deleted_time <= 1800:  # Restore within 30 minutes
                age = record.get('Age', 'N/A')
                gender = record.get('Gender', 'N/A')
                bmi = record.get('BMI', 'N/A')
                children = record.get('Children', 'N/A')
                smoker = "Yes" if record.get('Smoker') == "Yes" else "No"
                region = record.get('Region', 'N/A')
                charges = record.get('Charges', 'N/A')
                insurance_id = record.get('InsuranceID', 'N/A')
                restore_tree.insert('', 'end',
                values=(age, gender, bmi, children, smoker, region, charges, insurance_id))
    # Update the Treeview to reflect the changes
    restore_tree.update()
    # Restore selected records
    def restore_selected_records():
        selected_items = restore_tree.selection()
        if not selected_items:
            return
        # Restore selected records
        for item in selected_items:
            values = restore_tree.item(item, 'values')
            insurance_id = values[7]  # InsuranceID
            deleted_record = mydb["InsuranceInfo2"].find_one({"InsuranceID": int(insurance_id)})
            mycol.insert_one(deleted_record)
            mydb["InsuranceInfo2"].delete_one({"InsuranceID": int(insurance_id)})
        # Close the restore window and refresh the main GUI table
        restore_window.destroy()
        refresh_table()
        # Restore all records if all records are selected
        if len(selected_items) == len(restore_tree.get_children()):
            restore_all_records()

    def restore_all_records():
        deleted_records = mydb["InsuranceInfo2"].find({})
        for record in deleted_records:
            if "_deleted_at" in record:
                deleted_time = record["_deleted_at"]
                current_time = time.time()
                if current_time - deleted_time <= 1800:  # Restore within 30 minutes
                    mycol.insert_one(record)
                    mydb["InsuranceInfo2"].delete_one({"InsuranceID": record["InsuranceID"]})

        # Close the restore window and refresh the main GUI table
        restore_window.destroy()
        refresh_table()

    # Restore Button
    restore_button = ttk.Button(restore_window, text="Restore Selected Records", command=restore_selected_records)
    restore_button.grid(row=2, column=1, padx=10, pady=10)

    # Restore All Button
    restore_all_button = ttk.Button(restore_window, text="Restore All Records", command=restore_all_records)
    restore_all_button.grid(row=2, column=0, padx=10, pady=10)

# Update the restore function to open the new window
def restore():
    deleted_records = mydb["InsuranceInfo2"].find({})
    restore_deleted_records()
    refresh_table()

  # Clear the selected_items list after deletion
def refresh_table(cursor=None):
    tree.delete(*tree.get_children())
    if cursor is None:
        cursor = mycol.find({}).sort("InsuranceID", pymongo.ASCENDING)

    for text_fromDB in cursor:
        Age = str(text_fromDB.get('Age', 'N/A'))
        Gender = str(text_fromDB.get('Gender', 'N/A'))
        BMI = str(text_fromDB.get('BMI', 'N/A'))
        Children = str(text_fromDB.get('Children', 'N/A'))
        Smoker = "Yes" if text_fromDB.get('Smoker') == "Yes" else "No"  # Fix for displaying "Yes" or "No"
        Region = str(text_fromDB.get('Region', 'N/A'))
        Charges = str(text_fromDB.get('Charges', 'N/A'))
        InsuranceID = str(text_fromDB.get('InsuranceID', 'N/A'))
        tree.insert('', 'end', values=(Age, Gender, BMI, Children, Smoker, Region, Charges, InsuranceID))

# Connect to the MongoDB and set the collection
myclient = pymongo.MongoClient('mongodb://127.0.0.1:27017')
mydb = myclient['InsuranceML2']
mycol = mydb['InsuranceInfo3']
def save_or_update():
    selected_item = tree.selection()

    # Check if any of the required fields are empty
    if not age_var.get() or not bmi_var.get() or not children_var.get():
        messagebox.showerror('Error', 'Please fill in all the required fields.')
        return

    if selected_item:
        # Update an existing record
        update_record()
    else:
        # Insert a new record
        save()

# Create the main GUI window
root = tk.Tk()
root.title('Insurance Charges Prediction')
root.geometry("1200x400")

# Treeview for displaying the table
tree = ttk.Treeview(root, columns=('Age', 'Gender', 'BMI', 'Children', 'Smoker', 'Region', 'Charges', 'InsuranceID'))
tree.heading('#0', text='Index')
tree.heading('Age', text='Age')
tree.heading('Gender', text='Gender')
tree.heading('BMI', text='BMI')
tree.heading('Children', text='Children')
tree.heading('Smoker', text='Smoker')
tree.heading('Region', text='Region')
tree.heading('Charges', text='Charges')
tree.heading('InsuranceID', text='InsuranceID')

tree.column('#0', width=50)
tree.column('Age', width=50)
tree.column('Gender', width=100)
tree.column('BMI', width=100)
tree.column('Children', width=80)
tree.column('Smoker', width=80)
tree.column('Region', width=100)
tree.column('Charges', width=100)
tree.column('InsuranceID', width=100)

tree.grid(row=6, column=2, rowspan=4, columnspan=2, padx=10, pady=10)
tree.bind('<ButtonRelease-1>', callback)
def save():
    global mycol
    # Check if any of the required fields are empty
    if age_var.get() == "" or bmi_var.get() == "" or children_var.get() == "":
        messagebox.showerror('Error', 'Please fill in all the required fields.')
        return

    # Predict charges and set the charges_prediction value
    predicted_charges = predict_insurance_charges()
    charges_prediction.set(predicted_charges)

    # Check if the charges prediction value is a valid float
    charges_prediction_value = charges_prediction.get()

    try:
        charges_prediction_float = float(charges_prediction_value)
    except ValueError:
        messagebox.showerror('Error', 'Invalid charges prediction value. Please enter a valid number.')
        return

    r = messagebox.askokcancel('Save Record?', 'Do you want to save this record?')

    if r:
        newid = mycol.count_documents({})
        if newid != 0:
            newid = mycol.find_one(sort=[('InsuranceID', -1)])['InsuranceID']
        id = newid + 1
        ID_label2.set(id)

        # Get the string values from the StringVar objects
        age_value = age_var.get()
        sex_value = sex_var.get()
        bmi_value = bmi_var.get()
        children_value = children_var.get()
        smoker_value = smoker_var.get()
        region_value = region_var.get()
        charges_prediction_float = float(charges_prediction_value)
        charges_prediction_float = round(charges_prediction_float, 3)
        mydict = {
            "InsuranceID": int(ID_label2.get()),
            'Age': int(age_value),
            'Gender': sex_value,
            'BMI': float(bmi_value),
            'Children': int(children_value),
            'Smoker': smoker_value,
            'Region': region_value,
            'Charges': charges_prediction_float  # Use the validated float value
        }
        x = mycol.insert_one(mydict)
        refresh_table()

        # Clear the input fields
        age_var.set("")
        sex_var.set("Female")
        bmi_var.set("")
        children_var.set("")
        smoker_var.set("Yes")
        region_var.set("Southeast")
        charges_prediction.set("")
def safe_cast(value, to_type, default=None):
    try:
        return to_type(value)
    except (ValueError, TypeError):
        return default
def search_records():
    query = search_var.get()
    column = search_column_var.get()  # Get the selected column from the dropdown menu
    cursor = None

    if query and column:
        query_obj = {}

        if column == "Charges":
            try:
                query_obj[column] = {"$gte": float(query)}
            except ValueError:
                # Invalid input, so just return
                return
        elif column == "Age" or column == "Children" or column == "InsuranceID" or column == "BMI":
            try:
                query_obj[column] = int(query)
            except ValueError:
                # Invalid input, so just return
                return
        elif column == "Gender" or column == "Smoker":
            query_obj[column] = query.capitalize()  # Convert to lowercase for case-insensitive search
        elif column == "Region":
            pattern = re.compile(re.escape(query), re.IGNORECASE)
            query_obj[column] = pattern
        cursor = mycol.find(query_obj)
    refresh_table(cursor)
def update_record():
    global mycol, tree
    selected_item = tree.selection()
    if selected_item:
        item = selected_item[0]
        values = tree.item(item, 'values')
        insurance_id = safe_cast(values[7], int)  # InsuranceID
        # Check if any of the required fields are empty
        if not age_var.get() or not bmi_var.get() or not children_var.get():
            messagebox.showerror('Error', 'Please fill in all the required fields.')
            return
        r = messagebox.askokcancel("Update?", "Do you want to update this record?")
        if r:
            # Get the updated values from the input fields
            age_value = safe_cast(age_var.get(), int)
            sex_value = sex_var.get()
            bmi_value = safe_cast(bmi_var.get(), float)
            children_value = safe_cast(children_var.get(), int)
            smoker_value = smoker_var.get()
            region_value = region_var.get()
            # Predict new charges
            predicted_charges = predict_insurance_charges()
            # Update the selected record in the database
            myquery = {"InsuranceID": insurance_id}
            new_values = {
                "$set": {
                    'Age': age_value,
                    'Gender': sex_value,
                    'BMI': bmi_value,
                    'Children': children_value,
                    'Smoker': smoker_value,
                    'Region': region_value,
                    'Charges': predicted_charges  # Update the charges value with predicted charges
                }
            }
            mycol.update_one(myquery, new_values)
            # Update the selected record in the table
            tree.item(item, values=(
            age_value, sex_value, bmi_value, children_value, smoker_value, region_value, predicted_charges,
            insurance_id))

"""Data Collection & Analysis"""

# loading the data from the csv file to a Pandas DataFrame
insurance_dataset = pd.read_csv('/Users/xcite/Downloads/insurance.csv')

"""Categorical features:
*   Sex
*   Smoker
*   Region
"""
# Checking for missing values
insurance_dataset.isnull().sum()

"""Data Pre-processing

Encoding the Categorical features
"""

# encoding sex column
insurance_dataset.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)

# encoding 'smoker'column
insurance_dataset.replace({'smoker': {'no': 0, 'yes': 1}}, inplace=True)

# encoding 'region' column
insurance_dataset.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)

"""Splitting the Features and Target"""

X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']

print(X)

print(Y)

"""Splitting the data into Training and Testing data"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

"""Model Training

Random Forest Regression
"""
# loading the Random Forest Regressor model
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
pk.dump(regressor, open('model.pkl', 'wb'))
model = pk.load(open('model.pkl', 'rb'))
model.fit(X_train, Y_train)

"""Model Evaluation"""

# prediction on training data
training_data_prediction = model.predict(X_train)

# R squared value
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R squared value (Training): ', r2_train)

# prediction on test data
test_data_prediction = model.predict(X_test)

# R squared value
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R squared value (Test): ', r2_test)

# GUI implementation for prediction
def predict_insurance_charges():
    age = safe_cast(age_var.get(), int)
    sex = 1 if sex_var.get() == "Female" else 0
    bmi = safe_cast(bmi_var.get(), float)
    children = safe_cast(children_var.get(), int)
    smoker = 1 if smoker_var.get() == "Yes" else 0
    region = region_var.get()

    # Check if any of the required fields are empty
    region_mapping = {
        'Southeast': 0,
        'Southwest': 1,
        'Northeast': 2,
        'Northwest': 3
    }

    region_encoded = region_mapping.get(region, 0)  # Default to Southeast if region not recognized

    input_data = np.array([age, sex, bmi, children, smoker, region_encoded]).reshape(1, -1)

    prediction = model.predict(input_data)
    prediction_label.config(text=f'The insurance cost is USD: {prediction[0]:.3f}')

    charges_prediction.set(prediction[0])
    return prediction[0]

# StringVar to hold predicted charges
charges_prediction = tk.StringVar()
# InsuranceID

ID_label2 = tk.StringVar()

# Age
age_var = tk.StringVar()
age_label = ttk.Label(root, text='Age:')
age_label.grid(row=0, column=0)
age_entry = ttk.Entry(root, textvariable=age_var)
age_entry.grid(row=0, column=1)

# Gender
sex_var = tk.StringVar(value="Female")
sex_label = ttk.Label(root, text='Gender:')
sex_label.grid(row=1, column=0)
sex_combobox = ttk.Combobox(root, textvariable=sex_var, values=["Female", "Male"])
sex_combobox.grid(row=1, column=1)

# BMI
bmi_var = tk.StringVar()
bmi_label = ttk.Label(root, text='BMI:')
bmi_label.grid(row=2, column=0)
bmi_entry = ttk.Entry(root, textvariable=bmi_var)
bmi_entry.grid(row=2, column=1)

# Children
children_var = tk.StringVar()
children_label = ttk.Label(root, text='Children:')
children_label.grid(row=3, column=0)
children_entry = ttk.Entry(root, textvariable=children_var)
children_entry.grid(row=3, column=1)

# Smoker
smoker_var = tk.StringVar(value="Yes")
smoker_label = ttk.Label(root, text='Smoker:')
smoker_label.grid(row=4, column=0)
smoker_combobox = ttk.Combobox(root, textvariable=smoker_var, values=["Yes", "No"])
smoker_combobox.grid(row=4, column=1)

# Region
region_var = tk.StringVar(value="Southeast")
region_label = ttk.Label(root, text='Region:')
region_label.grid(row=5, column=0)
region_combobox = ttk.Combobox(root, textvariable=region_var,
values=["Southeast", "Southwest", "Northeast", "Northwest"])
region_combobox.grid(row=5, column=1)

#Search Bar
search_var = tk.StringVar()

search_entry = ttk.Entry(root, textvariable=search_var)
search_entry.grid(row=6, column=1, padx=5, pady=5)


# Search button
search_button = ttk.Button(root, text="Search", command=search_records)
search_button.grid(row=7, column=1, padx=5, pady=5)

# List of column names for search
search_columns = ["Age", "Gender", "BMI", "Children", "Smoker", "Region", "Charges", "InsuranceID"]

# StringVar to hold selected search column
search_column_var = tk.StringVar(value=search_columns[0])

# Search Column Label
search_column_combobox = ttk.Combobox(root, textvariable=search_column_var, values=search_columns)
search_column_combobox.grid(row=6, column=0, padx=5, pady=5)

# Predict Button
predict_button = ttk.Button(root, text="Predict", command=predict_insurance_charges)
predict_button.grid(row=7, column=0, padx=5, pady=5)

# Prediction Label
prediction_label = ttk.Label(root, text="")
prediction_label.grid(row=10, column=0, columnspan=1)

# Enter data into DB
enter_button = ttk.Button(root, text="Enter", command=save)
enter_button.grid(row=8, column=0, padx=5, pady=5)

# Bind the save() function to the Enter button
enter_button.configure(command=save)

# delete data in DB
delete_button = ttk.Button(root, text="Delete", command=delete)
delete_button.grid(row=9, column=0, padx=5, pady=5)
delete_button.configure(command=delete)

restore_button = ttk.Button(root, text="Restore", command=restore)
restore_button.grid(row=9, column=1, padx=5, pady=5)
restore_button.configure(command=restore)

# Update data in DB
update_button = ttk.Button(root, text="Update", command=save_or_update)
update_button.grid(row=8, column=1, padx=5, pady=5)

# Bind the callback() function to the Treeview selection event
tree.bind('<ButtonRelease-1>', callback)

# display the table in the GUI
refresh_table()
root.mainloop()