from datetime import datetime
from typing import Callable, List, Type, TypeVar, Union
import argparse

from .logger import get_logger
from .utils import parse_args

logger = get_logger()
_TArgsClass = TypeVar('_TArgsClass')
_T = TypeVar('_T')
NoneType = type(None)


def get_main(
    args_class: Type[_TArgsClass], llm_x: Callable[[_TArgsClass], _T]
) -> Callable[[Union[List[str], _TArgsClass, NoneType]], _T]:

    def x_main(argv: Union[List[str], _TArgsClass, NoneType] = None,
               **kwargs) -> _T:
        logger.info(
            f'Start time of running main: {datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}'
        )
        if not isinstance(argv, (list, tuple, NoneType)):
            args, remaining_argv = argv, []
        else:
            args, remaining_argv = parse_args(args_class, argv)
        if len(remaining_argv) > 0:
            if getattr(args, 'ignore_args_error', False):
                logger.warning(f'remaining_argv: {remaining_argv}')
            elif '--agma_gradient_accumulation_steps' in remaining_argv:
                parser = argparse.ArgumentParser()
                parser.add_argument('--agma_gradient_accumulation_steps', type=int)
                new_args, remaining_argv = parser.parse_known_args(remaining_argv)
                # merge new_args and args
                for k, v in vars(new_args).items():
                    setattr(args, k, v)
                    
                if len(remaining_argv) > 0:  
                     raise ValueError(f'remaining_argv: {remaining_argv}')
            else:
                raise ValueError(f'remaining_argv: {remaining_argv}')
        result = llm_x(args, **kwargs)
        logger.info(
            f'End time of running main: {datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}'
        )
        return result

    return x_main
