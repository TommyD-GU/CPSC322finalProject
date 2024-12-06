import importlib
import utils
import mypytable
importlib.reload(mypytable)
from mypytable import MyPyTable
import mysklearn.myknnclassifier as myknnclassifier

mush_data = MyPyTable()
mush_data.load_from_file('/home/CPSC322finalProject/new_mushroom_cleaned.csv')