for x in `ls |grep mlclass`
do cat "$x/submit.m" |grep "\.m'" |sed -n "s/[^']\+'\([^']*\)'.*/$x\/\1/gp"
done;
