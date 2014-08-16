what=$1
/usr/bin/awk --assign=R="${what}" -F" "  ' 
BEGIN { 
        print "*** ADDING ***" 
      } 
    { 
        for ( i=1; i<=NF; i++ ) {
            if ( NF == 1 ) {
                total += $1
                print ++x":", $1, ">>>", total, "<<<"
            }
        }
    } 
    END { 
            print x, "entries", "totaling:", total
        }
' $2
