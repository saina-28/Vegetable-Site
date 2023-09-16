//not secure 
const auth = (req, res, next) => {
    try{
        if(req.cookies.userid)
        {
            req.userid = req.cookies.userid
            console.log("Authenticated link ^")
            next()
        }
        else{ 
            // res.status(401).json({ message : "Unauthorized User"})
            res.redirect('/')
        }
    }
    catch(error)
    {
        console.log(error);
        // res.status(401).json({ message : "Unauthorized User"})
        res.redirect('/')
    }
}

module.exports = auth;