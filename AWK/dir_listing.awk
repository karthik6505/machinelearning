# 4 drwxrwxr-x   2 nrm nrm    4096 Jul 30 22:39 7f
what=$1
/usr/bin/awk --assign=R="${what}" -F" "  ' 
BEGIN { 
    print "*** SEARCHING FOR" , R, "***" 
    command = "sort -u -k 6"
    lineno = 100000
} 
    { 
        lineno++
        if ( NR % 100000000000000 == 0 ) {
            close(command)
            print ""
            print ""
            print ""
            print ""
            command = "sort -u -k 6"
            print "" | command
        }

        if ( NF == 1 ) {
            dirname=$0
            sub(":","/", dirname)
        }

        if ( NF > 4 ) {
            if ( $10 ~ R ) { 
                dates[$7 $8 lineno] = $10
                printf "%+06d %s %s \t %s blocks \t %s%s\n", ++x, $7, $8, $1, dirname, dates[$7 $8 lineno] | command
            }
        }
    } 
END { 
    close(command)
    # for ( date in dates ) printf "%s \t %s\n", date, dates[date]
    print "REGEX", R, "matched", x, "records"
}
' -
