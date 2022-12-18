import _Differentiation

extension ShapedArray: Differentiable where Scalar: Differentiable & AdditiveArithmetic {
    public typealias TangentVector = Self
}

extension ShapedArray: AdditiveArithmetic where Scalar: Differentiable & AdditiveArithmetic {
    @differentiable(reverse)
    public static func + (_ lhs: Self, _ rhs: Self) -> Self {
        var result = lhs
        result += rhs
        return result
    }

    @differentiable(reverse)
    public static func += (_ lhs: inout Self, _ rhs: Self) {
        if lhs._isScalarZero {
            lhs = rhs
        } else if !rhs._isScalarZero {
            for i in 0..<lhs.buffer.count {
                lhs.buffer[i] += rhs.buffer[i]
            }
        }
    }

    @derivative(of:+=)
    @usableFromInline
    static func vjpPlusEquals(_ lhs: inout Self, _ rhs: Self) -> (
        value: (), pullback: (inout Self) -> Self
    ) {
        lhs += rhs
        func pullback(_ v: inout Self) -> Self {
            return v
        }
        return ((), pullback)
    }

    @differentiable(reverse)
    public static func - (lhs: ShapedArray<Scalar>, rhs: ShapedArray<Scalar>) -> ShapedArray<Scalar> {
        var result = lhs
        result -= rhs
        return result
    }

    @differentiable(reverse)
    public static func -= (_ lhs: inout Self, _ rhs: Self) {
        if lhs._isScalarZero {
            lhs = rhs
        } else if !rhs._isScalarZero {
            for i in 0..<lhs.buffer.count {
                lhs.buffer[i] -= rhs.buffer[i]
            }
        }
    }

    @derivative(of:-=)
    @usableFromInline
    static func vjpMinusEquals(_ lhs: inout Self, _ rhs: Self) -> (
        value: (), pullback: (inout Self) -> Self
    ) {
        lhs -= rhs
        func pullback(_ v: inout Self) -> Self {
            return Self.zero - v
        }
        return ((), pullback)
    }

    public static var zero: Self {
        var zeroVal = Self.init(repeating: Scalar.zero, shape: [1])
        zeroVal._isScalarZero = true
        return zeroVal
    }

}
