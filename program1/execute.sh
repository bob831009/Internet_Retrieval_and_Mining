#!/bin/bash
# Put your command below to execute your program.
# Replace "./my-program" with the command that can execute your program.
# Remember to preserve " $@" at the end, which will be the program options we give you.
python Get_Doc_Len.py $@
python Get_VSM.py $@
python Query.py $@