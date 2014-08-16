ncols=$1
/usr/bin/awk --assign=N="${ncols}" -F" "  ' 
    BEGIN { 
        print "*** FINDING RECORDS HAVING LESS THAN", N, "COLUMNS ***" 
    } 
    { 
        if ( NF == N ) { 
            print ">>>", $0
            goodcnt++
        } else {
            print "<<<", $0
            defcnt++
        }
    } 
    END { 
            print goodcnt, defcnt, NR
        }
' -
