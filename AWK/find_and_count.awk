what=$1
/usr/bin/awk --assign=R="${what}" -F" "  ' 
BEGIN { 
        print "*** SEARCHING FOR" , R, "***" 
      } 
    { 
        if ( $0 ~ R ) 
            print ++x, ">>>", $0, "<<<"
    } 
    END { 
            print x, "matching lines total", R 
        }
' $2
