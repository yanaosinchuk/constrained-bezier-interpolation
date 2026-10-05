Attribute VB_Name = "modBezier"
Option Explicit

Public Type DoubleArray
  A() As Double
End Type

Public Type DoubleArrayArray
  AA() As DoubleArray
End Type

Public Function isRange(X As Variant) As Boolean
  isRange = (UCase(TypeName(X)) = "RANGE")
End Function

Public Function Dim2ToDAA(A As Variant) As DoubleArrayArray
  If isRange(A) Then A = A.value
  Dim result As DoubleArrayArray
  Dim i As Long, j As Long

  ReDim result.AA(1 To UBound(A, 1)) As DoubleArray

  For i = 1 To UBound(A, 1)
    Dim Row As DoubleArray
    ReDim Row.A(1 To UBound(A, 2))

    For j = 1 To UBound(A, 2)
      Row.A(j) = CDbl(A(i, j))
    Next j

    result.AA(i) = Row
  Next i
  Dim2ToDAA = result
End Function


' ============================================================
' COMPUTE CONTROL POINTS FOR ALL INTERVALS
' ============================================================

Private Function ComputeBoundedQuarticControlPoints( _
    ByRef xNodes() As Double, _
    ByRef yNodes() As Double, _
    ByRef means() As Double, _
    ByRef derivatives() As Double, _
    ByVal lowerBound As Double, _
    ByVal useUpperBound As Boolean, _
    ByVal upperBound As Double _
) As Double()

    Dim nIntervals As Long
    Dim controls() As Double
    
    Dim i As Long
    Dim h As Double
    Dim m As Double
    Dim d0 As Double, d1 As Double
    
    Dim P(0 To 4) As Double
    Dim internalSum As Double
    Dim P1Ref As Double, P3Ref As Double
    Dim projected() As Double
    
    nIntervals = UBound(xNodes) - LBound(xNodes)
    ReDim controls(0 To nIntervals - 1, 0 To 4)
    
    For i = 0 To nIntervals - 1
        
        h = xNodes(i + 1) - xNodes(i)
       
        m = means(i)
        
        d0 = derivatives(i)
        d1 = derivatives(i + 1)
        
        P(0) = yNodes(i)
        P(4) = yNodes(i + 1)
        
        ' Exact mean condition:
        '
        '   (P0 + P1 + P2 + P3 + P4) / 5 = mean
        '
        ' Therefore:
        '
        '   P1 + P2 + P3 = 5 * mean - P0 - P4
        internalSum = 5# * m - P(0) - P(4)
        
        ' Preferred derivative-based controls:
        '
        '   B'(left)  = 4(P1 - P0) / h
        '   B'(right) = 4(P4 - P3) / h
        P1Ref = P(0) + 0.25 * h * d0
        P3Ref = P(4) - 0.25 * h * d1
        
        projected = ProjectP1P3ToBoundedMeanFeasibleSet( _
            P1Ref, P3Ref, internalSum, _
            lowerBound, useUpperBound, upperBound, i _
        )
        
        P(1) = projected(0)
        P(3) = projected(1)
        P(2) = internalSum - P(1) - P(3)
        
        controls(i, 0) = P(0)
        controls(i, 1) = P(1)
        controls(i, 2) = P(2)
        controls(i, 3) = P(3)
        controls(i, 4) = P(4)
        
        Call CheckControlPointBounds(controls, i, lowerBound, useUpperBound, upperBound)
        
    Next i
    
    ComputeBoundedQuarticControlPoints = controls

End Function


' ============================================================
' PROJECTION OF P1 AND P3
' ============================================================

Private Function ProjectP1P3ToBoundedMeanFeasibleSet( _
    ByVal targetP1 As Double, _
    ByVal targetP3 As Double, _
    ByVal internalSum As Double, _
    ByVal lowerBound As Double, _
    ByVal useUpperBound As Boolean, _
    ByVal upperBound As Double, _
    ByVal intervalIndex As Long _
) As Double()

    Dim lb As Double, ub As Double
    Dim sumLow As Double, sumHigh As Double
    Dim minPossibleSum As Double, maxPossibleSum As Double
    Dim feasibleSumLow As Double, feasibleSumHigh As Double
    
    Dim bestP1 As Double, bestP3 As Double
    Dim bestDist As Double
    Dim hasCandidate As Boolean
    
    Dim candidateP1 As Double, candidateP3 As Double
    Dim fixedP1 As Double, fixedP3 As Double, fixedSum As Double
    Dim p1Low As Double, p1High As Double
    Dim p3Low As Double, p3High As Double
    Dim p1Star As Double
    
    Dim result(0 To 1) As Double
    
    lb = lowerBound
    
    If useUpperBound Then
        ub = upperBound
    Else
        ub = 1E+307
    End If
    
    If useUpperBound Then
        sumLow = internalSum - upperBound
    Else
        sumLow = -1E+307
    End If
    
    sumHigh = internalSum - lowerBound
    
    minPossibleSum = 2# * lb
    maxPossibleSum = 2# * ub
    
    feasibleSumLow = WorksheetFunction.Max(sumLow, minPossibleSum)
    feasibleSumHigh = WorksheetFunction.Min(sumHigh, maxPossibleSum)
    
    If feasibleSumLow > feasibleSumHigh Then
        Err.Raise vbObjectError + 1001, , _
            "Infeasible bounded quartic Bezier segment on interval " & intervalIndex & _
            ". Endpoint values, exact mean and bounds cannot be satisfied simultaneously."
    End If
    
    hasCandidate = False
    bestDist = 1E+307
    
    
    ' Candidate 1: target itself
    Call TryAddProjectionCandidate( _
        targetP1, targetP3, targetP1, targetP3, internalSum, _
        lb, useUpperBound, ub, bestP1, bestP3, bestDist, hasCandidate _
    )
    
    
    ' Boundary P1 = lowerBound
    fixedP1 = lb
    p3Low = lb
    p3High = ub
    
    If sumLow > -1E+307 Then p3Low = WorksheetFunction.Max(p3Low, sumLow - fixedP1)
    If sumHigh < 1E+307 Then p3High = WorksheetFunction.Min(p3High, sumHigh - fixedP1)
    
    If p3Low <= p3High Then
        candidateP3 = ClipD(targetP3, p3Low, p3High)
        Call TryAddProjectionCandidate( _
            fixedP1, candidateP3, targetP1, targetP3, internalSum, _
            lb, useUpperBound, ub, bestP1, bestP3, bestDist, hasCandidate _
        )
    End If
    
    
    ' Boundary P1 = upperBound
    If useUpperBound Then
        fixedP1 = ub
        p3Low = lb
        p3High = ub
        
        If sumLow > -1E+307 Then p3Low = WorksheetFunction.Max(p3Low, sumLow - fixedP1)
        If sumHigh < 1E+307 Then p3High = WorksheetFunction.Min(p3High, sumHigh - fixedP1)
        
        If p3Low <= p3High Then
            candidateP3 = ClipD(targetP3, p3Low, p3High)
            Call TryAddProjectionCandidate( _
                fixedP1, candidateP3, targetP1, targetP3, internalSum, _
                lb, useUpperBound, ub, bestP1, bestP3, bestDist, hasCandidate _
            )
        End If
    End If
    
    
    ' Boundary P3 = lowerBound
    fixedP3 = lb
    p1Low = lb
    p1High = ub
    
    If sumLow > -1E+307 Then p1Low = WorksheetFunction.Max(p1Low, sumLow - fixedP3)
    If sumHigh < 1E+307 Then p1High = WorksheetFunction.Min(p1High, sumHigh - fixedP3)
    
    If p1Low <= p1High Then
        candidateP1 = ClipD(targetP1, p1Low, p1High)
        Call TryAddProjectionCandidate( _
            candidateP1, fixedP3, targetP1, targetP3, internalSum, _
            lb, useUpperBound, ub, bestP1, bestP3, bestDist, hasCandidate _
        )
    End If
    
    
    ' Boundary P3 = upperBound
    If useUpperBound Then
        fixedP3 = ub
        p1Low = lb
        p1High = ub
        
        If sumLow > -1E+307 Then p1Low = WorksheetFunction.Max(p1Low, sumLow - fixedP3)
        If sumHigh < 1E+307 Then p1High = WorksheetFunction.Min(p1High, sumHigh - fixedP3)
        
        If p1Low <= p1High Then
            candidateP1 = ClipD(targetP1, p1Low, p1High)
            Call TryAddProjectionCandidate( _
                candidateP1, fixedP3, targetP1, targetP3, internalSum, _
                lb, useUpperBound, ub, bestP1, bestP3, bestDist, hasCandidate _
            )
        End If
    End If
    
    
    ' Boundary P1 + P3 = sumLow
    If sumLow > -1E+307 Then
        
        fixedSum = sumLow
        p1Star = 0.5 * (targetP1 - targetP3 + fixedSum)
        
        p1Low = lb
        p1High = ub
        
        p1Low = WorksheetFunction.Max(p1Low, fixedSum - ub)
        p1High = WorksheetFunction.Min(p1High, fixedSum - lb)
        
        If p1Low <= p1High Then
            candidateP1 = ClipD(p1Star, p1Low, p1High)
            candidateP3 = fixedSum - candidateP1
            
            Call TryAddProjectionCandidate( _
                candidateP1, candidateP3, targetP1, targetP3, internalSum, _
                lb, useUpperBound, ub, bestP1, bestP3, bestDist, hasCandidate _
            )
        End If
        
    End If
    
    
    ' Boundary P1 + P3 = sumHigh
    If sumHigh < 1E+307 Then
        
        fixedSum = sumHigh
        p1Star = 0.5 * (targetP1 - targetP3 + fixedSum)
        
        p1Low = lb
        p1High = ub
        
        p1Low = WorksheetFunction.Max(p1Low, fixedSum - ub)
        p1High = WorksheetFunction.Min(p1High, fixedSum - lb)
        
        If p1Low <= p1High Then
            candidateP1 = ClipD(p1Star, p1Low, p1High)
            candidateP3 = fixedSum - candidateP1
            
            Call TryAddProjectionCandidate( _
                candidateP1, candidateP3, targetP1, targetP3, internalSum, _
                lb, useUpperBound, ub, bestP1, bestP3, bestDist, hasCandidate _
            )
        End If
        
    End If
    
    
    If Not hasCandidate Then
        Err.Raise vbObjectError + 1002, , _
            "Projection failed on interval " & intervalIndex & _
            ". No feasible candidate was found."
    End If
    
    result(0) = bestP1
    result(1) = bestP3
    
    ProjectP1P3ToBoundedMeanFeasibleSet = result

End Function


Private Sub TryAddProjectionCandidate( _
    ByVal P1 As Double, _
    ByVal P3 As Double, _
    ByVal targetP1 As Double, _
    ByVal targetP3 As Double, _
    ByVal internalSum As Double, _
    ByVal lb As Double, _
    ByVal useUpperBound As Boolean, _
    ByVal ub As Double, _
    ByRef bestP1 As Double, _
    ByRef bestP3 As Double, _
    ByRef bestDist As Double, _
    ByRef hasCandidate As Boolean _
)

    Dim P2 As Double
    Dim dist As Double
    
    P2 = internalSum - P1 - P3
    
    If P1 < lb - 0.000000000001 Then Exit Sub
    If P3 < lb - 0.000000000001 Then Exit Sub
    If P2 < lb - 0.000000000001 Then Exit Sub
    
    If useUpperBound Then
        If P1 > ub + 0.000000000001 Then Exit Sub
        If P3 > ub + 0.000000000001 Then Exit Sub
        If P2 > ub + 0.000000000001 Then Exit Sub
    End If
    
    dist = (P1 - targetP1) ^ 2 + (P3 - targetP3) ^ 2
    
    If Not hasCandidate Or dist < bestDist Then
        bestDist = dist
        bestP1 = P1
        bestP3 = P3
        hasCandidate = True
    End If

End Sub

' ============================================================
' DERIVATIVE ESTIMATION: PCHIP-LIKE METHOD
' ============================================================
Function GetColumn(arr As Variant, Optional n As Long = 1) As Variant
    Dim result() As Variant
    Dim i As Long
    Dim lb As Long
    Dim ub As Long
    Dim P As Long
    
    lb = LBound(arr, 1)
    ub = UBound(arr, 1)
    P = ub - lb
    
    ReDim result(0 To P) As Variant
    
    For i = 0 To P
      result(i) = arr(lb + i, n)
    Next i
    
    GetColumn = result
End Function

Public Function EstimatePchipDerivatives( _
    xNodes As Variant, _
    yNodes As Variant _
) As Variant

    Dim n As Long
    Dim i As Long
    
    Dim h() As Double
    Dim delta() As Double
    Dim d() As Double
    
    Dim leftSlope As Double
    Dim rightSlope As Double
    Dim w1 As Double
    Dim w2 As Double
    
    If TypeName(xNodes) = "Range" Then
      xNodes = GetColumn(xNodes.value)
    End If
    
    If TypeName(yNodes) = "Range" Then
      yNodes = GetColumn(yNodes.value)
    End If
    
    n = UBound(xNodes, 1) - LBound(xNodes, 1) + 1
    
    ReDim h(0 To n - 2)
    ReDim delta(0 To n - 2)
    ReDim d(0 To n - 1)
    
    If n = 2 Then
        d(0) = (yNodes(1) - yNodes(0)) / (xNodes(1) - xNodes(0))
        d(1) = d(0)
        EstimatePchipDerivatives = d
        Exit Function
    End If
    
    For i = 0 To n - 2
        h(i) = xNodes(i + 1) - xNodes(i)
        delta(i) = (yNodes(i + 1) - yNodes(i)) / h(i)
    Next i
    
    ' Interior points
    For i = 1 To n - 2
        
        leftSlope = delta(i - 1)
        rightSlope = delta(i)
        
        If leftSlope = 0# Or rightSlope = 0# Or Sgn(leftSlope) <> Sgn(rightSlope) Then
            d(i) = 0#
        Else
            w1 = 2# * h(i) + h(i - 1)
            w2 = h(i) + 2# * h(i - 1)
            d(i) = (w1 + w2) / (w1 / leftSlope + w2 / rightSlope)
        End If
        
    Next i
    
    ' Endpoints
    d(0) = PchipEndpointSlope(h(0), h(1), delta(0), delta(1))
    d(n - 1) = PchipEndpointSlope(h(n - 2), h(n - 3), delta(n - 2), delta(n - 3))
    
    EstimatePchipDerivatives = d

End Function


Private Function PchipEndpointSlope( _
    ByVal h0 As Double, _
    ByVal h1 As Double, _
    ByVal delta0 As Double, _
    ByVal delta1 As Double _
) As Double

    Dim d As Double
    
    d = ((2# * h0 + h1) * delta0 - h0 * delta1) / (h0 + h1)
    
    If d = 0# Then
        PchipEndpointSlope = 0#
        Exit Function
    End If
    
    If Sgn(d) <> Sgn(delta0) Then
        PchipEndpointSlope = 0#
        Exit Function
    End If
    
    If Sgn(delta0) <> Sgn(delta1) And Abs(d) > Abs(3# * delta0) Then
        PchipEndpointSlope = 3# * delta0
        Exit Function
    End If
    
    PchipEndpointSlope = d

End Function


' ============================================================
' EVALUATION OF QUARTIC BEZIER - de-Casteljau-Verfahren
' ============================================================
Public Function EvaluateQuarticBezierSegment( _
    ByVal P0 As Double, _
    ByVal P1 As Double, _
    ByVal P2 As Double, _
    ByVal P3 As Double, _
    ByVal P4 As Double, _
    ByVal t As Double _
) As Double

    Dim u As Double

    If t < 0# Then t = 0#
    If t > 1# Then t = 1#

    u = 1# - t

    ' 1. Stufe
    P0 = u * P0 + t * P1
    P1 = u * P1 + t * P2
    P2 = u * P2 + t * P3
    P3 = u * P3 + t * P4

    ' 2. Stufe
    P0 = u * P0 + t * P1
    P1 = u * P1 + t * P2
    P2 = u * P2 + t * P3

    ' 3. Stufe
    P0 = u * P0 + t * P1
    P1 = u * P1 + t * P2

    ' 4. Stufe
    P0 = u * P0 + t * P1

    EvaluateQuarticBezierSegment = P0

End Function

Private Function Bezier1D_Internal(ByVal t As Double, P() As Double) As Double
    Dim lo As Long, hi As Long
    Dim i As Long, k As Long
    Dim u As Double
    
    lo = LBound(P)
    hi = UBound(P)
    u = 1 - t
    
    For k = 1 To (hi - lo)
        For i = lo To hi - k
            P(i) = u * P(i) + t * P(i + 1)
        Next i
    Next k
    
    Bezier1D_Internal = P(lo)
End Function


Public Function RangeToLinearDouble(ByVal P As Variant) As Double()
    Dim v As Variant
    Dim d() As Double
    Dim R As Long, c As Long
    Dim rMax As Long, cMax As Long
    Dim idx As Long
    
    ' Range.Value ? 2D Variant SAFEARRAY (immer 1-basiert)
    v = P.value
    
    ' UBounds nur einmal auslesen
    rMax = UBound(v, 1)
    cMax = UBound(v, 2)
    
    ' 1D-Array dimensionieren
    ReDim d(1 To rMax * cMax)
    
    ' Linearisation
    idx = 1
    For R = 1 To rMax
        For c = 1 To cMax
            d(idx) = CDbl(v(R, c))
            idx = idx + 1
        Next c
    Next R
    
    RangeToLinearDouble = d
End Function


Public Function EvaluatePiecewiseQuarticBezierAtX( _
    ByRef xNodes() As Double, _
    ByRef controls() As Double, _
    ByVal xValue As Double _
) As Double

    Dim nIntervals As Long
    Dim i As Long
    Dim t As Double
    
    nIntervals = UBound(xNodes) - LBound(xNodes)
    
    If xValue <= xNodes(0) Then
        EvaluatePiecewiseQuarticBezierAtX = controls(0, 0)
        Exit Function
    End If
    
    If xValue >= xNodes(nIntervals) Then
        EvaluatePiecewiseQuarticBezierAtX = controls(nIntervals - 1, 4)
        Exit Function
    End If
    
    For i = 0 To nIntervals - 1
        
        If xValue >= xNodes(i) And xValue <= xNodes(i + 1) Then
            
            t = (xValue - xNodes(i)) / (xNodes(i + 1) - xNodes(i))
            
            EvaluatePiecewiseQuarticBezierAtX = EvaluateQuarticBezierSegment( _
                controls(i, 0), controls(i, 1), controls(i, 2), _
                controls(i, 3), controls(i, 4), t _
            )
            
            Exit Function
            
        End If
        
    Next i
    
    Err.Raise vbObjectError + 1003, , "xValue is outside interpolation range."

End Function


' ============================================================
' VERIFICATION FUNCTIONS
' ============================================================

Public Function VerifyMaxMeanError( _
    ByRef controls() As Double, _
    ByRef means() As Double _
) As Double

    Dim i As Long
    Dim nIntervals As Long
    Dim segmentMean As Double
    Dim errValue As Double
    Dim maxErr As Double
    
    nIntervals = UBound(means) - LBound(means) + 1
    maxErr = 0#
    
    For i = 0 To nIntervals - 1
        
        segmentMean = (controls(i, 0) + controls(i, 1) + controls(i, 2) + controls(i, 3) + controls(i, 4)) / 5#
        errValue = Abs(segmentMean - means(i))
        
        If errValue > maxErr Then maxErr = errValue
        
    Next i
    
    VerifyMaxMeanError = maxErr

End Function


Public Function VerifyMinControlValue( _
    ByRef controls() As Double _
) As Double

    Dim i As Long
    Dim j As Long
    Dim minValue As Double
    
    minValue = controls(0, 0)
    
    For i = LBound(controls, 1) To UBound(controls, 1)
        For j = LBound(controls, 2) To UBound(controls, 2)
            If controls(i, j) < minValue Then minValue = controls(i, j)
        Next j
    Next i
    
    VerifyMinControlValue = minValue

End Function


Public Function ComputeControlPolygonRoughness( _
    ByRef controls() As Double _
) As Double

    Dim i As Long
    Dim roughness As Double
    Dim secondDiff As Double
    
    roughness = 0#
    
    For i = LBound(controls, 1) To UBound(controls, 1)
        
        secondDiff = controls(i, 0) - 2# * controls(i, 1) + controls(i, 2)
        roughness = roughness + secondDiff ^ 2
        
        secondDiff = controls(i, 1) - 2# * controls(i, 2) + controls(i, 3)
        roughness = roughness + secondDiff ^ 2
        
        secondDiff = controls(i, 2) - 2# * controls(i, 3) + controls(i, 4)
        roughness = roughness + secondDiff ^ 2
        
    Next i
    
    ComputeControlPolygonRoughness = roughness

End Function


' ============================================================
' VALIDATION
' ============================================================

Private Sub ValidateInputArrays( _
    ByRef xNodes() As Double, _
    ByRef yNodes() As Double, _
    ByRef means() As Double _
)

    Dim nX As Long
    Dim nY As Long
    Dim nM As Long
    Dim i As Long
    
    nX = UBound(xNodes) - LBound(xNodes) + 1
    nY = UBound(yNodes) - LBound(yNodes) + 1
    nM = UBound(means) - LBound(means) + 1
    
    If nX <> nY Then
        Err.Raise vbObjectError + 2001, , "xNodes and yNodes must have the same length."
    End If
    
    If nX < 2 Then
        Err.Raise vbObjectError + 2002, , "At least two support points are required."
    End If
    
    If nM <> nX - 1 Then
        Err.Raise vbObjectError + 2003, , "means must have length len(xNodes) - 1."
    End If
    
    For i = 0 To nX - 2
        If xNodes(i + 1) <= xNodes(i) Then
            Err.Raise vbObjectError + 2004, , "xNodes must be strictly increasing."
        End If
    Next i

End Sub


Private Sub CheckEndpointBounds( _
    ByRef yNodes() As Double, _
    ByVal lowerBound As Double, _
    ByVal useUpperBound As Boolean, _
    ByVal upperBound As Double _
)

    Dim i As Long
    
    For i = LBound(yNodes) To UBound(yNodes)
        
        If yNodes(i) < lowerBound Then
            Err.Raise vbObjectError + 2005, , _
                "Support values violate lowerBound. Exact bounded interpolation is impossible."
        End If
        
        If useUpperBound Then
            If yNodes(i) > upperBound Then
                Err.Raise vbObjectError + 2006, , _
                    "Support values violate upperBound. Exact bounded interpolation is impossible."
            End If
        End If
        
    Next i

End Sub


Private Sub CheckControlPointBounds( _
    ByRef controls() As Double, _
    ByVal intervalIndex As Long, _
    ByVal lowerBound As Double, _
    ByVal useUpperBound As Boolean, _
    ByVal upperBound As Double _
)

    Dim jj As Long
    Dim value As Double
    
    For jj = 0 To 4
        
        value = controls(intervalIndex, jj)
        
        If value < lowerBound - 0.0000000001 Then
            Err.Raise vbObjectError + 2007, , _
                "Projection error on interval " & intervalIndex & _
                ". A control point is below lowerBound."
        End If
        
        If useUpperBound Then
            If value > upperBound + 0.0000000001 Then
                Err.Raise vbObjectError + 2008, , _
                    "Projection error on interval " & intervalIndex & _
                    ". A control point is above upperBound."
            End If
        End If
        
    Next jj

End Sub


' ============================================================
' SMALL HELPER FUNCTIONS
' ============================================================

Private Function ClipD( _
    ByVal value As Double, _
    ByVal lowerValue As Double, _
    ByVal upperValue As Double _
) As Double

    If value < lowerValue Then
        ClipD = lowerValue
    ElseIf value > upperValue Then
        ClipD = upperValue
    Else
        ClipD = value
    End If

End Function

' ============================================================
' EXCEL-CELL USER DEFINED FUNCTIONS
' ============================================================

Public Function BEZIER_CONTROL_POINT( _
    ByVal xLeft As Double, _
    ByVal xRight As Double, _
    ByVal trueMean As Double, _
    ByVal dLeft As Double, _
    ByVal dRight As Double, _
    ByVal yLeft As Double, _
    ByVal yRight As Double, _
    Optional ByVal lowerBound As Double = 0#, _
    Optional ByVal useUpperBound As Boolean = False, _
    Optional ByVal upperBound As Double = 0# _
) As Variant

    Dim h As Double
    Dim P(0 To 0, 0 To 4) As Double
    Dim internalSum As Double
    Dim P1Ref As Double, P3Ref As Double
    Dim projected() As Double

'    If whichP < 0 Or whichP > 4 Then
'        BEZIER_CONTROL_POINT = CVErr(xlErrValue)
'        Exit Function
'    End If

    h = xRight - xLeft

    If h <= 0# Then
        BEZIER_CONTROL_POINT = CVErr(xlErrValue)
        Exit Function
    End If

    P(0, 0) = yLeft
    P(0, 4) = yRight

    internalSum = 5# * trueMean - P(0, 0) - P(0, 4)

    P1Ref = P(0, 0) + (h / 4#) * dLeft
    P3Ref = P(0, 4) - (h / 4#) * dRight

    projected = ProjectP1P3ToBoundedMeanFeasibleSet( _
        P1Ref, P3Ref, internalSum, lowerBound, useUpperBound, upperBound, 0 _
    )

    P(0, 1) = projected(0)
    P(0, 3) = projected(1)
    P(0, 2) = internalSum - P(0, 1) - P(0, 3)
    
    BEZIER_CONTROL_POINT = P

End Function

Public Function BEZIER_VALUE_AT_X( _
    xValue As Double, _
    x_left As Variant, _
    x_right As Variant, _
    ControlPoints As Variant _
) As Double

    Dim i As Long
    Dim nIntervals As Long
    Dim xLeft As Double, xRight As Double
    Dim t As Double

    nIntervals = UBound(x_left.value, 1) - LBound(x_left.value, 1) + 1

'    If controlRange.Rows.Count <> nIntervals Or controlRange.Columns.Count <> 5 Then
'        BEZIER_VALUE_AT_X = CVErr(xlErrValue)
'        Exit Function
'    End If

'    If xValue <= CDbl(intervalRange.Cells(1, 1).value) Then
'        BEZIER_VALUE_AT_X = CDbl(controlRange.Cells(1, 1).value)
'        Exit Function
'    End If

'    If xValue >= CDbl(intervalRange.Cells(nIntervals, 2).value) Then
'        BEZIER_VALUE_AT_X = CDbl(controlRange.Cells(nIntervals, 5).value)
'        Exit Function
'    End If

    For i = 1 To nIntervals

        xLeft = x_left(i, 1)
        xRight = x_right(i, 1)

        If xValue >= xLeft And xValue <= xRight Then

            t = (xValue - xLeft) / (xRight - xLeft)

            BEZIER_VALUE_AT_X = EvaluateQuarticBezierSegment( _
                ControlPoints(i, 1), _
                ControlPoints(i, 2), _
                ControlPoints(i, 3), _
                ControlPoints(i, 4), _
                ControlPoints(i, 5), _
                t _
            )

            Exit Function

        End If

    Next i

    BEZIER_VALUE_AT_X = CVErr(xlErrNA)

End Function


Public Function BEZIER_MAX_MEAN_ERROR( _
    ByVal controls As Variant, _
    ByVal means As Variant _
) As Double

    Dim i As Long
    Dim segmentMean As Double
    Dim errValue As Double
    Dim maxErr As Double

    maxErr = 0#

    For i = LBound(controls.value) To UBound(controls.value)

        segmentMean = ( _
            controls(i, 1) + _
            controls(i, 2) + _
            controls(i, 3) + _
            controls(i, 4) + _
            controls(i, 5) _
        ) / 5#

        errValue = Abs(segmentMean - means(i, 1))
        If errValue > maxErr Then maxErr = errValue

    Next i

    BEZIER_MAX_MEAN_ERROR = maxErr

End Function


Public Function BEZIER_TOTAL_ROUGHNESS( _
    ByVal controls As Variant _
) As Double

    Dim i As Long
    Dim roughness As Double
    Dim secondDiff As Double

'    If controlRange.Columns.Count <> 5 Then
'        BEZIER_TOTAL_ROUGHNESS = CVErr(xlErrValue)
'        Exit Function
'    End If

    roughness = 0#

    For i = LBound(controls.value) To UBound(controls.value)

        secondDiff = controls(i, 1) - 2# * controls(i, 2) + controls(i, 3)
        roughness = roughness + secondDiff ^ 2

        secondDiff = controls(i, 2) - 2# * controls(i, 3) + controls(i, 4)
        roughness = roughness + secondDiff ^ 2

        secondDiff = controls(i, 3) - 2# * controls(i, 4) + controls(i, 5)
        roughness = roughness + secondDiff ^ 2

    Next i

    BEZIER_TOTAL_ROUGHNESS = roughness

End Function

