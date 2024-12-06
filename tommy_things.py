import importlib
import utils
import mysklearn.mypytable as mypytable
importlib.reload(mypytable)
from mysklearn.mypytable import MyPyTable
import mysklearn.myknnclassifier as myknnclassifier

mush_data = MyPyTable()
mush_data.load_from_file('/home/CPSC322finalProject/new_mushroom_cleaned.csv')