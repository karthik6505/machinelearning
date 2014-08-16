ncol=$1
what=$2
/usr/bin/awk --assign N="${ncol}" --assign W="${what}" -F","  ' 
    BEGIN { 
        print "*** SELECT CSV RECORDS FOR WHICH COLUMN", N, "EQUALS", W, "***" 
    } 
    { 
            split( $0, arr, "," ); 
            if( arr[N] == W ) { 
                cnt++;
                print ">>>", $0 
            } else {
                notmatch++
            }
    }
    END { 
            print cnt, notmatch, NR
        }
' -
