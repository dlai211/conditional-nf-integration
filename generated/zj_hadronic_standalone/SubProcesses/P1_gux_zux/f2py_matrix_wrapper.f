      SUBROUTINE PY_M2_SMATRIXHEL(P,HEL,ANS)
      IMPLICIT NONE
C
C CONSTANT
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=4)
      INTEGER                 NCOMB         
      PARAMETER (             NCOMB=24)
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: HEL
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)  

C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL),ANS
	  INTEGER HEL

      CALL M2_SMATRIXHEL(P,HEL,ANS)
	  END

      SUBROUTINE PY_M2_SMATRIX(P,ANS)
C  
C 
C MadGraph5_aMC@NLO StandAlone Version
C 
C Returns amplitude squared summed/avg over colors
c and helicities
c for the point in phase space P(0:3,NEXTERNAL)
C  
C Process: g u~ > z u~ WEIGHTED<=3 @1
C Process: g c~ > z c~ WEIGHTED<=3 @1
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=4)
	  INTEGER    NINITIAL 
      PARAMETER (NINITIAL=2)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL),ANS
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
      call M2_SMATRIX(P,ANS)
	  END
       
       
      REAL*8 FUNCTION PY_M2_MATRIX(P,NHEL,IC)
C  
C
C Returns amplitude squared -- no average over initial state/symmetry factor
c for the point with external lines W(0:6,NEXTERNAL)
C  
C Process: g u~ > z u~ WEIGHTED<=3 @1
C Process: g c~ > z c~ WEIGHTED<=3 @1
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=4)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
C
C  FUNCTIONS
C
      real*8 M2_MATRIX
      PY_M2_MATRIX = M2_MATRIX(P,NHEL,IC)
      END

      SUBROUTINE PY_M2_GET_value(P, ALPHAS, NHEL ,ANS)
      IMPLICIT NONE   
C
C CONSTANT
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=4)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL),ANS
      INTEGER NHEL
      DOUBLE PRECISION ALPHAS 
CF2PY INTENT(OUT) :: ANS  
CF2PY INTENT(IN) :: NHEL   
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL) 
CF2PY INTENT(IN) :: ALPHAS
      call M2_GET_value(P, ALPHAS, NHEL ,ANS)
      return 
      end

      SUBROUTINE PY_M2_INITIALISEMODEL(PATH)
C     ROUTINE FOR F2PY to read the benchmark point.    
      IMPLICIT NONE   
      CHARACTER*512 PATH
CF2PY INTENT(IN) :: PATH 
      call setpara(PATH)  !first call to setup the paramaters    
      return 
      end      

      LOGICAL FUNCTION PY_M2_IS_BORN_HEL_SELECTED(HELID)
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=4)
C
C     ARGUMENTS
C
      INTEGER HELID
      LOGICAL M2_IS_BORN_HEL_SELECTED
      PY_M2_IS_BORN_HEL_SELECTED = M2_IS_BORN_HEL_SELECTED(HELID)
      RETURN
      END
