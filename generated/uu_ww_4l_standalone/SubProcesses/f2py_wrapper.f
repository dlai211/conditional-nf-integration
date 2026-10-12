
C     PY ((21, 21), (-6, 6)) : (21, 21, 6, -6) # M0_ 1
C     PY ((2, -2), (-6, 6)) : (2, -2, 6, -6) # M1_ 1
C     PY ((4, -4), (-6, 6)) : (4, -4, 6, -6) # M1_ 1
C     PY ((1, -1), (-6, 6)) : (1, -1, 6, -6) # M1_ 1
C     PY ((3, -3), (-6, 6)) : (3, -3, 6, -6) # M1_ 1
      SUBROUTINE SMATRIXHEL(PDGS, PROCID, NPDG, P, ALPHAS, SCALE2,
     $  NHEL, ANS)
      IMPLICIT NONE
C     ALPHAS is given at scale2 (SHOULD be different of 0 for loop
C     induced, ignore for LO)  

CF2PY double precision, intent(in), dimension(0:3,npdg) :: p
CF2PY integer, intent(in), dimension(npdg) :: pdgs
CF2PY integer, intent(in):: procid
CF2PY integer, intent(in) :: npdg
CF2PY double precision, intent(out) :: ANS
CF2PY double precision, intent(in) :: ALPHAS
CF2PY double precision, intent(in) :: SCALE2
      INTEGER PDGS(*)
      INTEGER NPDG, NHEL, PROCID
      DOUBLE PRECISION P(*)
      DOUBLE PRECISION ANS, ALPHAS,SCALE2

      CALL F77_SMATRIXHEL(PDGS, PROCID, NPDG, P, ALPHAS, SCALE2, NHEL,
     $  ANS)

      RETURN
      END

      SUBROUTINE INITIALISE(PATH)
C     ROUTINE FOR F2PY to read the benchmark point.
      IMPLICIT NONE
      CHARACTER*512 PATH
CF2PY INTENT(IN) :: PATH
      CALL SETPARA(PATH)  !first call to setup the paramaters
      RETURN
      END


      SUBROUTINE CHANGE_PARA(NAME, VALUE)
      IMPLICIT NONE
CF2PY intent(in) :: name
CF2PY intent(in) :: value

      CHARACTER*512 NAME
      DOUBLE PRECISION VALUE
      CALL F77_CHANGE_PARA(NAME, VALUE)

      RETURN
      END

      SUBROUTINE UPDATE_ALL_COUP()
      IMPLICIT NONE
      CALL F77_UPDATE_ALL_COUP()
      RETURN
      END


      SUBROUTINE GET_PDG_ORDER(PDG, ALLPROC)
      IMPLICIT NONE
CF2PY INTEGER, intent(out) :: PDG(1,6)
CF2PY INTEGER, intent(out) :: ALLPROC(1)
      INTEGER PDG(1,6)
      INTEGER ALLPROC(1)
      CALL F77_GET_PDG_ORDER(PDG, ALLPROC)
      RETURN
      END

      SUBROUTINE GET_PREFIX(PREFIX)
      IMPLICIT NONE
CF2PY CHARACTER*20, intent(out) :: prefix(1)
      CHARACTER*20 PREFIX(1)
      CALL F77_GET_PREFIX(PREFIX)
      RETURN
      END

      SUBROUTINE SET_FIXED_EXTRA_SCALE(NEW_VALUE)
      IMPLICIT NONE
CF2PY logical, intent(in) :: new_value
      LOGICAL NEW_VALUE
      CALL F77_SET_FIXED_EXTRA_SCALE(NEW_VALUE)
      RETURN
      END

      SUBROUTINE SET_MUE_OVER_REF(NEW_VALUE)
      IMPLICIT NONE
CF2PY double precision, intent(in) :: new_value
      DOUBLE PRECISION NEW_VALUE

      CALL F77_SET_MUE_OVER_REF(NEW_VALUE)
      RETURN
      END

      SUBROUTINE SET_MUE_REF_FIXED(NEW_VALUE)
      IMPLICIT NONE
CF2PY double precision, intent(in) :: new_value
      DOUBLE PRECISION NEW_VALUE

      CALL F77_SET_MUE_REF_FIXED(NEW_VALUE)
      RETURN
      END


      SUBROUTINE SET_MAXJETFLAVOR(NEW_VALUE)
      IMPLICIT NONE
CF2PY integer, intent(in) :: new_value
      INTEGER NEW_VALUE
      CALL F77_SET_MAXJETFLAVOR(NEW_VALUE)

      RETURN
      END


      SUBROUTINE SET_ASMZ(NEW_VALUE)
      IMPLICIT NONE
CF2PY double precision, intent(in) :: new_value
      DOUBLE PRECISION NEW_VALUE
      CALL F77_SET_ASMZ(NEW_VALUE)
      RETURN
      END

      SUBROUTINE SET_NLOOP(NEW_VALUE)
      IMPLICIT NONE
CF2PY integer, intent(in) :: new_value
      INTEGER NEW_VALUE
      CALL F77_SET_NLOOP(NEW_VALUE)
      RETURN
      END



      SUBROUTINE M0_GET_NHEL_ENTRY()
      INTEGER M0_NHEL(6,64)
      COMMON/M0_PROCESS_NHEL/M0_NHEL
      CALL F77_M0_GET_NHEL_ENTRY(M0_NHEL)

      RETURN
      END



