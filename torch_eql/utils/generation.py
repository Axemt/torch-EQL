from typing import List, Callable, Tuple, Literal
import torch

XYPair = Tuple[torch.Tensor, torch.Tensor]

class TestFNS:

    TestFNSignature = Callable[[torch.Tensor], torch.Tensor]

    @staticmethod
    def FG1a(X: torch.Tensor) -> torch.Tensor:
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        return (1/3.0)*(torch.sin(x1*torch.pi)+torch.sin(2*torch.pi*x2+torch.pi/8)+x2-x3*x4)
     
    @staticmethod
    def Ackley(X: torch.Tensor) -> torch.Tensor:
        
        x = X[:, 0]
        y = X[:, 1]

        return -20 * torch.exp( -0.2 * torch.sqrt(0.5 * (torch.pow(x, 2) + torch.pow(y, 2)))) - torch.exp( 0.5 * (torch.cos(2*torch.pi*x) + torch.cos(2*torch.pi*y) ) ) + torch.e + 20

    @staticmethod
    def Easom(X: torch.Tensor) -> torch.Tensor:

        x = X[:, 0]
        y = X[:, 1]

        return - torch.cos(x)*torch.cos(y)*torch.exp( -((x - torch.pi)**2 + (y - torch.pi) ** 2) )

    @staticmethod
    def RossenbrockNDFactory(N: int, a: int = 1, b: int = 100) -> TestFNSignature:

        def _RossenbrockND(X: torch.Tensor) -> torch.Tensor:
            
            batch = X.shape[0]
            res = torch.zeros(batch).to(X.device)

            for i in range(N-1):
                term1 = (b * (X[:, i + 1] - X[:, i] ** 2) ** 2)
                term2 = ((a - X[:, i]) ** 2)
                res += term1 + term2

            return res

        return _RossenbrockND
    
    @staticmethod
    def Rossenbrock(X: torch.Tensor, **kwargs) -> torch.Tensor:

        N = X.shape[-1]

        return TestFNS.RossenbrockNDFactory(N, **kwargs)(X)

    @staticmethod
    def RastriginFactory(N: int, A: int = 10) -> TestFNSignature:

        def _RastriginND(X: torch.Tensor) -> torch.Tensor:

            return A * N + torch.sum(( X**2 - A * torch.cos(2* torch.pi * X )), axis=-1)

        return _RastriginND
    
    @staticmethod
    def Rastrigin(X: torch.Tensor, **kwargs) -> torch.Tensor:

        N = X.shape[-1]
        return TestFNS.RastriginFactory(N, **kwargs)(X)
    
    @staticmethod
    def Beale(X: torch.Tensor) -> torch.Tensor:

        x = X[:, 0]
        y = X[:, 1]
        return (1.5 - x + x*y) ** 2 + (2.25 - x + (x*y) ** 2 ) ** 2 + (2.625 - x + (x*y) ** 3 ) ** 2
    
    @staticmethod
    def Matyas(X: torch.Tensor) -> torch.Tensor:

        x = X[:, 0]
        y = X[:, 1]

        return 0.26 * (x**2 + y**2) - 0.48 * x * y
    
    @staticmethod
    def Weierstrass(X: torch.Tensor, a: float = 0.5, b: float = 7.0, n_terms: int = 10) -> torch.Tensor:

        res = torch.zeros((X.shape[0]))

        for n in range(n_terms):
            t = a ** n * torch.cos( b ** n * torch.pi * X[:, 0] )
            res += t

        return res
    

    @staticmethod
    def Branin(
        X: torch.Tensor, 
        a: float = 1, 
        b: float = 5.1/(4*torch.pi**2), 
        c: float = 5/torch.pi, 
        r: float = 6, 
        s: float = 10,
        t: float = 1/ (8*torch.pi) 
    ) -> torch.Tensor:
        # From https://www.sfu.ca/~ssurjano/branin.html

        x1 = X[:, 0]
        x2 = X[:, 1]

        return a*(x2 - b*x1**2 + c*x1 - r) ** 2 + s*(t-1)*torch.cos(x1) + s
    
    @staticmethod
    def ChengSandu2010(X: torch.Tensor) -> torch.Tensor:
        # From https://www.sfu.ca/~ssurjano/chsan10.html

        x1 = X[:, 0]
        x2 = X[:, 1]

        return torch.cos(x1 + x2) * torch.exp(x1*x2)

    #@staticmethod
    #def SulfurModel(X: torch.Tensor, S0: float = 1361.0, A: float = 5e14) -> torch.Tensor:
    #    # From https://www.sfu.ca/~ssurjano/sulf.html
#
    #    Tr       = X[:, 1]
    #    Ac       = X[:, 2]
    #    Rs       = X[:, 3]
    #    beta_bar = X[:, 4]
    #    Psi_e    = X[:, 5]
    #    f_Psi_e  = X[:, 6]
    #    Q        = X[:, 7]
    #    Y        = X[:, 8]
    #    L        = X[:, 9]
#
    #    fact1 = (S0^2) * (1-Ac) * (Tr^2) * (1-Rs)^2 * beta_bar * Psi_e * f_Psi_e
    #    fact2 = 3*Q*Y*L / A
#
    #    return -1/2 * fact1 * fact2


TEST_FNS = {
    name.lower() : TestFNS.__getattribute__(TestFNS, name)
    for name in filter(
        lambda name: ('__' not in name) and ('Factory' not in name) and name != 'TestFNSignature',
        dir(TestFNS)
    )
}


def generate_sample(
    generator_fn: Callable[[torch.Tensor], torch.Tensor],
    fn_in_dims: int,
    interval: List[float] | float,
    density: int = int(1e3),
    noise_std: float= 0.01,
    device: str = 'cpu'
    ) -> XYPair:

    if isinstance(interval, (float, int)):
        interval = [-interval, interval]

    x = torch.distributions.uniform.Uniform(
        low=interval[0], high=interval[1],
    ).sample(
        (density,fn_in_dims),
    ).to(device)

    y = generator_fn(x).to(device)

    y_noise = torch.empty(y.shape, device=device).normal_(std=noise_std)

    return (x, y + y_noise)

def generate_disjoint_test(
    generator_fn: Callable[[torch.Tensor], torch.Tensor],
    fn_in_dims: int,
    seen_interval: List[float] | float, 
    unseen_interval: List[float] | float,
    seen_density: int = int(1e3),
    unseen_density: int = int(1e4),
    noise_std: float = 0.01,
    device: str = 'cpu'
    ) -> Tuple[XYPair]:

    if isinstance(seen_interval, (float, int)):
        seen_interval = [-seen_interval, seen_interval]

    if isinstance(unseen_interval, (float, int)):
        unseen_interval = [-unseen_interval, unseen_interval]
        
    x_seens, y_seens = generate_sample(
        generator_fn=generator_fn,
        fn_in_dims=fn_in_dims,
        interval=seen_interval,
        density=seen_density,
        noise_std=noise_std,
        device=device
    )

    x_unseens_left, y_unseens_left = generate_sample(
        generator_fn=generator_fn,
        fn_in_dims=fn_in_dims,
        interval=[unseen_interval[0], seen_interval[0]],
        density=unseen_density//2,
        noise_std=noise_std,
        device=device
    )


    x_unseens_right, y_unseens_right = generate_sample(
        generator_fn=generator_fn,
        fn_in_dims=fn_in_dims,
        interval=[seen_interval[1], unseen_interval[1]],
        density=unseen_density//2,
        noise_std=noise_std,
        device=device
    )
    
    x_unseens = torch.concat([ x_unseens_left, x_unseens_right ])
    y_unseens = torch.concat([ y_unseens_left, y_unseens_right])

    return (x_seens, y_seens), (x_unseens, y_unseens)
