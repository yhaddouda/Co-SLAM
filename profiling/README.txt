******************Scripts*********************************
1- gprof_automate_yh.sh:
    -> This file generates the visuialization graph using gprof2dot tool; it contains the whole pipeline from training the model and generating a cprofile extention file
        to converting this file to a .dot extention and then to a png readable image. 
    -> Different variables can be tweaked in the .sh script to get different results depending on the desired results.


2- pstats_automate_yh.sh:
    -> This file outputs the cprofile results to a txt file; it contains the whole pipeline from training the model and generating a cprofile extention file
        to processing the file to only keep the files in the desired nerfstudio directory and sorting the files by "tottime".
    -> Different variables can be tweaked in the .sh script to get different results and/or do other processing tasks.

3- py-spy record -o coslam.svg -d 60 -s -- python coslam.py --config configs/Replica/office2.yaml
    -> **Not a script but a library** that creates a .svg file with the callstack that can be opened in a browser (especially useful for multiprocessing applications, where cprofile fails)
        it samples the callstack of processes by intervals, here it does that for 60 seconds, and 100 samples per second

4- System Monitoring
    -> Step 1: *in the first terminal: python advanced_system_monitor.py --duration 150 --interval 0.5 --output office2_experiment 
                *in parallel in a second terminal : python coslam_mp.py --config configs/Replica/office2.yaml
        The script will produce a .json and a .csv file with the date and hour, and .txt file with SUMMARY
        datasets and intervals and durations can be changed as suited

    -> Step 2: python advanced_system_visualizer.py office2_experiment_20250616_173101.json --temporal
        For a temporal representation of memory, cpus cores and GPU utilizaion (for other flags -h)
               python advanced_system_visualizer_frequency.py office2_experiment_20250616_173101.json 
        For a more frequency representation of the data


******************Usage*********************************
1- Make the scripts executable:
    chmod +x gprof_automate_yh.sh pstats_automate_yh.sh
2- Run the desired script:
    ./gprof_automate_yh.sh   or   ./pstats_automate_yh.sh




******************Requirements*********************************
1- gprof2dot
   graphviz
3- py-spy
4- pandas matplotlib seaborn numpy








