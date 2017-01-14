#!/bin/tcsh
#SBATCH --time=01:00:00
#SBATCH -N 1
#SBATCH -n 4
# additional information for allocated clusters
#SBATCH --account=dorval
#SBATCH --partition=ember

#################################################################
# This script executes several parallel jobs simultaneously.
# Number of jobs is equal to number of Slurm tasks (-n x)
# and must be equal to number of lines for WORKDIR and PROGRAM
# You have the option of using different scratches.
#################################################################

#################################################################
# Initial info, good for reporting problems
#################################################################
echo Beginning job on `hostname` on `date`
echo Job number $SLURM_JOB_ID
echo Running on nodes: $SLURM_NODELIST

#################################################################
# Define location (global or local)
# Note: when using 'local' your whole scratch directory space will 
# be removed before the job starts (to clean up left-overs from 
# previous crashed runs).
# Never change $SCRATCHDIR (below) into anything other then a real 
# /scratch/local space without also changing LOCATION to 
# 'global'. If you do, your whole work space will be lost!!!
#################################################################
# /scratch/kingspeak/serial
# /scratch/lonepeak/serial
# /scratch/telluride/serial
# /scratch/ibrix/chpc_gen
# /scratch/local
set LOCATION   = global/lustre


#################################################################
# Supply details of programs, directories, and files
# WORKDIR - data directory, unique for each job
# PROGRAM - the program to run, can be the same or unique for each run
# SCRATCHDIR - scratch directory for intermediate data storage
#              replace username with your username
#################################################################
# All Python scripts are in the same directory in this case
set WORKDIR    = ( $HOME/u0389528/spwr_network \
                   $HOME/u0389528/spwr_network \
                   $HOME/u0389528/spwr_network \
                   $HOME/u0389528/spwr_network )
set PROGRAM    = (python3 /uufs/chpc.utah.edu/common/home/u0389528/spwr_network/test1.py  \
                  python3 /uufs/chpc.utah.edu/common/home/u0389528/spwr_network/test2.py  \
                  python3 /uufs/chpc.utah.edu/common/home/u0389528/spwr_network/test3.py  \
                  python3 /uufs/chpc.utah.edu/common/home/u0389528/spwr_network/test4.py  )
set SCRATCHDIR = /scratch/$LOCATION/u0389528

module load anaconda # do I need this here? Or is it something I just do from the command line?

#################################################################
#################################################################
# The real stuff. Don't mess with this, or you are on your own...
################################################################
#################################################################

cd $SLURM_SUBMIT_DIR

#################################################################
# create script for a single run
#################################################################
cat << EOM > single_script.csh
#!/bin/tcsh
mkdir -p \$1
sync
wait
cp \$2/* \$1
sync
cd \$1
\$3 >& output
cp * \$2
sync
cd $HOME
rm -rf \$1
EOM

chmod u+x single_script.csh

#################################################################
# create Slurm multi-prog config file
#################################################################
rm -f multiprog.conf
@ JOBNR = 0
while ($JOBNR < $SLURM_NTASKS)
  @ JOBNR1 = $JOBNR + 1
  echo $JOBNR $SLURM_SUBMIT_DIR/single_script.csh $SCRATCHDIR/$SLURM_JOB_ID/$JOBNR1 $WORKDIR[$JOBNR1] $PROGRAM[$JOBNR1] >> multiprog.conf
  @ JOBNR = $JOBNR + 1
end

#################################################################
# run using srun --multi-prog
#################################################################
srun --multi-prog multiprog.conf

#################################################################
# Final info, good for reporting problems
#################################################################
echo You made it!
echo You have been running on the following processors:
echo Job completed on `date`.
