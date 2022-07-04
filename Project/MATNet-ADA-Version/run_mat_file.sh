#!/bin/bash
# or u can just type matlab in terminal after module loading it (module load matlab/R2021a) and then
# type the name of the function you want to run without .m extension. If it were to call any function, then
# it will do it nicely. 
matlab -nodisplay -nosplash -softwareopengl < "$1" > "$2"
echo "File Executed!!!"
echo "---------------------------------------------------------------------------------------------------"
