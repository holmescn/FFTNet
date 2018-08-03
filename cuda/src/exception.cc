#include <cstdio>
#include <cstring>
#include "cudnn.h"
#include "exception.h"

char cudnn::Exception::_buffer[4096] = { 0 };
char cuda::Exception::_buffer[4096] = { 0 };

cudnn::Exception&
cudnn::Exception::operator=(const cudnn::Exception &other)
{
    return *this = Exception(other);    
}

cudnn::Exception&
cudnn::Exception::operator=(cudnn::Exception &&other) noexcept
{
    this->_status = other._status;
    this->_file = other._file;
    this->_line = other._line;
    return *this;
}

const char* cudnn::Exception::what() const noexcept {
    sprintf(this->_buffer, "Error in %s line %lu: %s",
            this->_file, this->_line, cudnnGetErrorString(this->_status));
    return this->_buffer;
}


cuda::Exception&
cuda::Exception::operator=(const cuda::Exception &other)
{
    return *this = Exception(other);    
}

cuda::Exception&
cuda::Exception::operator=(cuda::Exception &&other) noexcept
{
    this->_status = other._status;
    this->_file = other._file;
    this->_line = other._line;
    return *this;
}

const char* cuda::Exception::what() const noexcept {
    sprintf(this->_buffer, "Error in %s line %lu: %s",
            this->_file, this->_line, cudaGetErrorString(this->_status));
    return this->_buffer;
}
